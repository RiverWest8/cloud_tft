

"""
Temporal Fusion Transformer (TFT) pipeline
==========================================

This script trains a single TFT model that jointly predicts:
  • realised volatility (quantile regression on an asinh‑transformed target)
  • the direction of the next period’s price move (binary classification)

It expects three parquet files:
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_train.parquet
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_val.parquet
    ▸ /Users/riverwest-gomila/Desktop/Data/CleanedData/universal_test.parquet

Required columns (exact names or common aliases will be auto‑detected):
    asset         : categorical asset identifier (aliases: symbol, ticker, instrument)
    Time          : timestamp (parsed to pandas datetime)
    realised_vol  : one‑period realised volatility target
    direction     : 0/1 label for next‑period price direction
plus any engineered numeric features

What the script does:
  1) Loads the parquet splits and standardises column names.
  2) Adds a per‑asset integer `time_idx` required by PyTorch‑Forecasting.
  3) Builds a `TimeSeriesDataSet` with **two targets**: ["realised_vol", "direction"].
     • Target normalisation:
         – realised_vol: `GroupNormalizer(..., transformation="asinh", scale_by_group=True)`
           (applies asinh in the normaliser, per asset)
         – direction: identity (no transform)
     • A per‑asset median `rv_scale` is also attached for **fallback decoding only**
       if a normaliser decode is unavailable.
  4) Fits a TemporalFusionTransformer with a dual head: 7 quantiles for vol and one
     logit for direction, using `LearnableMultiTaskLoss(AsymmetricQuantileLoss, LabelSmoothedBCE)`.
  5) Saves checkpoints, metrics, predictions, and quick feature‑importance summaries.

Adjust hyper‑parameters in the CONFIG section below.
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from typing import List
import json
import numpy as np
import pandas as pd
import lightning as pl
from lightning.pytorch import Trainer, seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ Added imports for new FI block ------------------

from pytorch_forecasting import (
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data import MultiNormalizer, TorchNormalizer

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

# --- Cloud / performance helpers ---
import fsspec
try:
    import gcsfs  # ensure GCS protocol is registered with fsspec
except Exception:
    gcsfs = None

# CUDA perf knobs / utils
import shutil
import argparse


# -----------------------------------------------------------------------
# Ensure a robust "identity" transformation for GroupNormalizer
# -----------------------------------------------------------------------
#
# Different versions of PyTorch‑Forecasting store transformations as a
# dictionary mapping *name* ➜ {"forward": fn, "inverse": fn}.  Some older
# releases omit "identity" entirely, which triggers a KeyError.  Other
# versions include it but as a bare function instead of a dict, which then
# breaks later when `.setdefault()` is called on it.  The logic below
# handles both cases safely.
#
identity_transform = {"forward": lambda x: x, "inverse": lambda x: x}

# If the key is missing OR is not in the expected dict format, patch it.
if (
    "identity" not in GroupNormalizer.TRANSFORMATIONS
    or not isinstance(GroupNormalizer.TRANSFORMATIONS["identity"], dict)
):
    GroupNormalizer.TRANSFORMATIONS["identity"] = identity_transform

# Some PF versions expose a separate `INVERSE_TRANSFORMATIONS` registry.
# Add the mapping only if the attribute exists.

if hasattr(GroupNormalizer, "INVERSE_TRANSFORMATIONS"):
    GroupNormalizer.INVERSE_TRANSFORMATIONS.setdefault("identity", lambda x: x)

# ---------------- Register custom asinh transformation ----------------
if (
    "asinh" not in GroupNormalizer.TRANSFORMATIONS
    or not isinstance(GroupNormalizer.TRANSFORMATIONS["asinh"], dict)
):
    GroupNormalizer.TRANSFORMATIONS["asinh"] = {
        "forward": lambda x: torch.asinh(x) if torch.is_tensor(x) else np.arcsinh(x),
        "inverse": lambda x: torch.sinh(x) if torch.is_tensor(x) else np.sinh(x),
    }
if hasattr(GroupNormalizer, "INVERSE_TRANSFORMATIONS"):
    GroupNormalizer.INVERSE_TRANSFORMATIONS.setdefault(
        "asinh",
        lambda x: torch.sinh(x) if torch.is_tensor(x) else np.sinh(x),
    )



# -----------------------------------------------------------------------
# Compatibility shim: older PyTorch‑Forecasting versions expose only
# `.inverse_transform()` but not `.decode()`.  Downstream code expects
# `.decode()`, so add an alias when missing.
# -----------------------------------------------------------------------
if not hasattr(GroupNormalizer, "decode"):
    def _gn_decode(self, y, group_ids=None, **kwargs):
        """
        Alias for `inverse_transform` to keep newer *and* older
        PyTorch‑Forecasting versions compatible with the same call‑site.

        The underlying `inverse_transform` API has changed a few times:
        ▸ Newer versions accept ``group_ids=`` keyword
        ▸ Some legacy variants want ``X=`` instead
        ▸ Very old releases implement the method but raise ``NotImplementedError``  
          (it was a placeholder).

        The cascading fall‑backs below try each signature in turn and, as a
        last resort, simply return the *input* unchanged so downstream code
        can continue without crashing.
        """
        try:
            # 1️⃣  Modern signature (>=0.10): accepts ``group_ids=``
            return self.inverse_transform(y, group_ids=group_ids, **kwargs)
        except (TypeError, NotImplementedError):
            try:
                # 2️⃣  Mid‑vintage signature: expects ``X=None`` instead
                return self.inverse_transform(y, X=None, **kwargs)
            except (TypeError, NotImplementedError):
                try:
                    # 3️⃣  Very early signature: just (y) positional
                    return self.inverse_transform(y)
                except (TypeError, NotImplementedError):
                    # 4️⃣  Ultimate fall‑back – give up on denorm, return y
                    return y

    GroupNormalizer.decode = _gn_decode



from pytorch_forecasting.metrics import QuantileLoss, MultiLoss



class LabelSmoothedBCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, pos_weight: float = 1.1):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(self, y_pred, target):
        target = target.float()
        target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            y_pred.squeeze(-1), target.squeeze(-1),
            pos_weight=self.pos_weight,
        )

class AsymmetricQuantileLoss(QuantileLoss):
    """
    Pinball (quantile) loss with an extra multiplier applied *only* when the
    prediction is BELOW the ground‑truth value (i.e. under‑prediction).
    Setting ``underestimation_factor`` > 1 makes the model pay a larger
    penalty for forecasts that are too low.
    """
    def __init__(self, quantiles, underestimation_factor: float = 10, **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)
        self.underestimation_factor = float(underestimation_factor)

    def loss_per_prediction(self, y_pred, target):
        import torch
        diff = target.unsqueeze(-1) - y_pred  # positive ⇒ under‑prediction
        q = torch.tensor(self.quantiles, device=y_pred.device).view(
            *([1] * (diff.ndim - 1)), -1
        )
        loss = torch.where(
            diff >= 0,
            self.underestimation_factor * q * diff,  # heavier penalty when under‑predicting
            (1 - q) * (-diff),                       # standard pinball for over‑prediction
        )
        return loss


class PerAssetMetrics(pl.Callback):
    """Collects per-asset predictions during validation and prints/saves metrics.
    Computes MAE, RMSE, MSE, QLIKE for realised_vol and Accuracy for direction.
    """
    def __init__(self, id_to_name: dict, vol_normalizer, max_print: int = 10):
        super().__init__()
        self.id_to_name = {int(k): str(v) for k, v in id_to_name.items()}
        self.vol_norm = vol_normalizer
        self.max_print = max_print
        self.reset()

    def reset(self):
        # device-resident accumulators (concatenate at epoch end)
        self._g_dev = []   # group ids per sample (device, flattened)
        self._yv_dev = []  # realised vol target (NORMALISED, device)
        self._pv_dev = []  # realised vol pred   (NORMALISED, device)
        self._yd_dev = []  # direction target (device)
        self._pd_dev = []  # direction pred logits/probs (device)
        self._t_dev = []   # decoder time_idx (device) if provided
        # cached final rows/overall from last epoch
        self._last_rows = None
        self._last_overall = None

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module):
        self.reset()

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0):
        # batch is (x, y, weight) from PF dataloader
        if not isinstance(batch, (list, tuple)):
            return
        x = batch[0]
        if not isinstance(x, dict):
            return
        groups = x.get("groups")
        dec_t = x.get("decoder_target")
        # optional time index for plotting/joining later
        dec_time = x.get("decoder_time_idx", None)
        if dec_time is None:
            # some PF versions may expose relative index or time via different keys; try a few
            dec_time = x.get("decoder_relative_idx", None)
        # also fetch explicit targets from batch[1] as a fallback
        y_batch = batch[1] if isinstance(batch, (list, tuple)) and len(batch) >= 2 else None
        if groups is None:
            return

        # groups can be a Tensor or a list[Tensor]; take the first if list
        groups_raw = groups[0] if isinstance(groups, (list, tuple)) else groups
        g = groups_raw
        # squeeze trailing singleton dims to get [B]
        while torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
            g = g.squeeze(-1)
        if not torch.is_tensor(g):
            return

        # --- Extract targets (try decoder_target first, else fall back to batch[1]) ---
        y_vol_t, y_dir_t = None, None
        if dec_t is not None:
            if torch.is_tensor(dec_t):
                y = dec_t
                if y.ndim == 3 and y.size(-1) == 1:
                    y = y[..., 0]  # → [B, n_targets]
                if y.ndim == 2 and y.size(1) >= 1:
                    y_vol_t = y[:, 0]
                    if y.size(1) > 1:
                        y_dir_t = y[:, 1]
            elif isinstance(dec_t, (list, tuple)) and len(dec_t) >= 1:
                y_vol_t = dec_t[0]
                if torch.is_tensor(y_vol_t):
                    if y_vol_t.ndim == 3 and y_vol_t.size(-1) == 1:
                        y_vol_t = y_vol_t[..., 0]
                    if y_vol_t.ndim == 2 and y_vol_t.size(-1) == 1:
                        y_vol_t = y_vol_t[:, 0]
                if len(dec_t) > 1 and torch.is_tensor(dec_t[1]):
                    y_dir_t = dec_t[1]
                    if y_dir_t.ndim == 3 and y_dir_t.size(-1) == 1:
                        y_dir_t = y_dir_t[..., 0]
                    if y_dir_t.ndim == 2 and y_dir_t.size(-1) == 1:
                        y_dir_t = y_dir_t[:, 0]
        # Fallback: PF sometimes provides targets in batch[1] as [B, pred_len, n_targets]
        if (y_vol_t is None or y_dir_t is None) and torch.is_tensor(y_batch):
            yb = y_batch
            if yb.ndim == 3 and yb.size(1) == 1:
                yb = yb[:, 0, :]
            if yb.ndim == 2 and yb.size(1) >= 1:
                if y_vol_t is None:
                    y_vol_t = yb[:, 0]
                if y_dir_t is None and yb.size(1) > 1:
                    y_dir_t = yb[:, 1]
        if y_vol_t is None:
            return
      
        # Forward pass to get predictions for this batch
        y_hat = pl_module(x)
        pred = getattr(y_hat, "prediction", y_hat)
        if isinstance(pred, dict) and "prediction" in pred:
            pred = pred["prediction"]

        def _extract_heads(pred):
            """Return (p_vol, p_dir) as 1D tensors [B] on DEVICE.
            Handles outputs as:
              • list/tuple: [vol(…,7), dir(…,7 or 1/2)]
              • tensor [B, 2, 7] (after our predict_step squeeze)
              • tensor [B, 1, 7] (vol only)
              • tensor [B, 7] (vol only)
            """
            import torch

            def _to_median_q(t):
                if t is None:
                    return None
                if t.ndim == 3 and t.size(1) == 1:
                    t = t.squeeze(1)  # [B,7]
                if t.ndim == 2 and t.size(-1) >= 1:
                    return t[:, t.size(-1) // 2]
                if t.ndim == 1:
                    return t
                return t.reshape(t.size(0), -1)[:, 0]

            def _to_dir_logit(t):
                if t is None:
                    return None
                if t.ndim == 3 and t.size(1) == 1:
                    t = t.squeeze(1)  # [B,7] or [B,1]
                # if replicated across 7, take middle slot; if single, squeeze
                if t.ndim == 2 and t.size(-1) >= 1:
                    return t[:, t.size(-1) // 2]
                if t.ndim == 1:
                    return t
                return t.reshape(t.size(0), -1)[:, 0]

            # Case 1: PF-style list/tuple per target
            if isinstance(pred, (list, tuple)):
                p_vol_t = pred[0]
                p_dir_t = pred[1] if len(pred) > 1 else None
                return _to_median_q(p_vol_t), _to_dir_logit(p_dir_t)

            # Case 2: our DualOutputModule returns a stacked tensor
            if torch.is_tensor(pred):
                t = pred
                # squeeze prediction-length dim if present → [B, 2, 7] or [B, 1, 7]
                if t.ndim == 4 and t.size(1) == 1:
                    t = t.squeeze(1)
                # Full two-head output: [B, 2, 7]
                if t.ndim == 3 and t.size(1) == 2:
                    vol = t[:, 0, :]  # [B,7]
                    d   = t[:, 1, :]  # [B,7] replicated direction logit
                    return _to_median_q(vol), _to_dir_logit(d)
                # Vol-only variants
                if t.ndim == 3 and t.size(1) == 1:
                    vol = t[:, 0, :]  # [B,7]
                    return _to_median_q(vol), None
                if t.ndim == 2 and t.size(-1) >= 1:
                    return _to_median_q(t), None

            # Fallback: treat as vol-only
            return _to_median_q(pred), None

        p_vol, p_dir = _extract_heads(pred)
        if p_vol is None:
            return

        # Store device tensors; no decode/CPU here
        L = g.shape[0]
        self._g_dev.append(g.reshape(L))
        self._yv_dev.append(y_vol_t.reshape(L))
        self._pv_dev.append(p_vol.reshape(L))
        # capture time index if available and shape-compatible
        if dec_time is not None and torch.is_tensor(dec_time):
            tvec = dec_time
            # squeeze to [B]
            while tvec.ndim > 1 and tvec.size(-1) == 1:
                tvec = tvec.squeeze(-1)
            if tvec.numel() >= L:
                self._t_dev.append(tvec.reshape(-1)[:L])

        if y_dir_t is not None and p_dir is not None:
            y_flat = y_dir_t.reshape(-1)
            p_flat = p_dir.reshape(-1)
            L2 = min(L, y_flat.numel(), p_flat.numel())
            if L2 > 0:
                self._yd_dev.append(y_flat[:L2])
                self._pd_dev.append(p_flat[:L2])

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._g_dev:
            return
        device = self._g_dev[0].device
        g = torch.cat(self._g_dev).to(device)
        yv = torch.cat(self._yv_dev).to(device)
        pv = torch.cat(self._pv_dev).to(device)
        yd = torch.cat(self._yd_dev).to(device) if self._yd_dev else None
        pd = torch.cat(self._pd_dev).to(device) if self._pd_dev else None

        # decode realised vol in one shot
        try:
            yv_dec_t = self.vol_norm.decode(yv.unsqueeze(-1), group_ids=g.unsqueeze(-1)).squeeze(-1)
        except Exception:
            yv_dec_t = yv
        try:
            pv_dec_t = self.vol_norm.decode(pv.unsqueeze(-1), group_ids=g.unsqueeze(-1)).squeeze(-1)
        except Exception:
            pv_dec_t = pv

        # move to CPU for numpy-style metrics
        g_cpu = g.detach().cpu()
        yv_dec_all = yv_dec_t.detach().cpu()
        pv_dec_all = pv_dec_t.detach().cpu()
        yd_cpu = yd.detach().cpu() if yd is not None else None
        pd_cpu = pd.detach().cpu() if pd is not None else None

        uniq = torch.unique(g_cpu)
        rows = []
        eps = 1e-8
        for gid in uniq.tolist():
            m = g_cpu == gid
            yvi = yv_dec_all[m]
            pvi = pv_dec_all[m]
            mae = (pvi - yvi).abs().mean().item()
            mse = ((pvi - yvi) ** 2).mean().item()
            rmse = mse ** 0.5
            sigma2_p = torch.clamp(pvi.abs(), min=eps) ** 2  # forecast variance
            sigma2_p = torch.clamp(sigma2_p, min=eps)
            sigma2   = torch.clamp(yvi.abs(), min=eps) ** 2  # realized variance
            ratio = sigma2 / sigma2_p  # true/forecast
            qlike = (ratio - torch.log(ratio) - 1.0).mean().item()
            acc = None
            if yd_cpu is not None and pd_cpu is not None and m.sum() > 0:
                ydi = yd_cpu[m]
                pdi = pd_cpu[m]
                try:
                    if torch.isfinite(pdi).any() and (pdi.min() < 0 or pdi.max() > 1):
                        pdi = torch.sigmoid(pdi)
                except Exception:
                    pdi = torch.sigmoid(pdi)
                acc = ((pdi >= 0.5).int() == ydi.int()).float().mean().item()
            name = self.id_to_name.get(int(gid), str(gid))
            rows.append((name, mae, rmse, mse, qlike, acc, int(m.sum().item())))

        overall_mae = (pv_dec_all - yv_dec_all).abs().mean().item()
        overall_mse = ((pv_dec_all - yv_dec_all) ** 2).mean().item()
        overall_rmse = overall_mse ** 0.5
        sigma2_p_all = torch.clamp(pv_dec_all.abs(), min=eps) ** 2  # forecast variance
        sigma2_p_all = torch.clamp(sigma2_p_all, min=eps)
        sigma2_all   = torch.clamp(yv_dec_all.abs(), min=eps) ** 2  # realized variance
        ratio_all    = sigma2_all / sigma2_p_all  # true/forecast
        overall_qlike = (ratio_all - torch.log(ratio_all) - 1.0).mean().item()

        # —— expose decoded metrics, but DO NOT override PF's native val_loss ——
        try:
            import torch as _torch
            # Direction BCE (logits-aware) over all collected val samples
            dir_bce = None
            if yd_cpu is not None and pd_cpu is not None and yd_cpu.numel() > 0 and pd_cpu.numel() > 0:
                yt = yd_cpu.float()
                pt = pd_cpu
                # if looks like logits, use BCEWithLogits; else plain BCE on probs
                if _torch.isfinite(pt).any() and (pt.min() < 0 or pt.max() > 1):
                    # label smoothing 0.1 (same as training)
                    yt_s = yt * 0.9 + 0.05
                    dir_bce = F.binary_cross_entropy_with_logits(pt, yt_s)
                else:
                    p = _torch.clamp(pt, 1e-7, 1 - 1e-7)
                    yt_s = yt * 0.9 + 0.05
                    dir_bce = F.binary_cross_entropy(p, yt_s)
                dir_bce = float(dir_bce.item())
            # Composite validation metric (decoded): MAE + RMSE (direction excluded)
            comp_val = float(overall_mae) + float(overall_rmse)
            trainer.callback_metrics["val_mae_dec"] = _torch.tensor(float(overall_mae))
            trainer.callback_metrics["val_rmse_dec"] = _torch.tensor(float(overall_rmse))
            trainer.callback_metrics["val_dir_bce"] = _torch.tensor(float(dir_bce)) if dir_bce is not None else _torch.tensor(float('nan'))
            trainer.callback_metrics["val_loss_decoded"] = _torch.tensor(comp_val)
        except Exception:
            pass

        self._last_rows = sorted(rows, key=lambda r: r[-1], reverse=True)
        self._last_overall = {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "mse": overall_mse,
            "qlike": overall_qlike,
            "yd": yd_cpu,
            "pd": pd_cpu,
        }
        # ---- concise per-epoch validation metrics printout ----
        try:
            epoch_num = int(getattr(trainer, "current_epoch", -1)) + 1
        except Exception:
            epoch_num = None
        try:
            ov = self._last_overall
            msg = f"[VAL EPOCH {epoch_num}] MAE={ov['mae']:.6f} RMSE={ov['rmse']:.6f} QLIKE={ov['qlike']:.6f}"
            # quick direction accuracy snapshot if available
            dir_acc = None
            if yd is not None and pd is not None and yd.numel() > 0 and pd.numel() > 0:
                p1 = pd
                try:
                    if torch.isfinite(p1).any() and (p1.min() < 0 or p1.max() > 1):
                        p1 = torch.sigmoid(p1)
                except Exception:
                    p1 = torch.sigmoid(p1)
                p1 = torch.clamp(p1, 0.0, 1.0)
                dir_acc = ((p1 >= 0.5).int() == yd.int()).float().mean().item()
            if dir_acc is not None:
                msg += f" | DIR_ACC={dir_acc:.3f}"
            print(msg)
        except Exception:
            pass

    @torch.no_grad()
    def on_fit_end(self, trainer, pl_module):
        if self._last_rows is None or self._last_overall is None:
            return
        rows = self._last_rows
        overall = self._last_overall
        print("\nOverall decoded metrics (final):")
        print(f"MAE: {overall['mae']:.6f} | RMSE: {overall['rmse']:.6f} | MSE: {overall['mse']:.6f} | QLIKE: {overall['qlike']:.6f}")

        print("\nPer-asset validation metrics (top by samples):")
        print("asset | MAE | RMSE | MSE | QLIKE | ACC | N")
        for r in rows[: self.max_print]:
            acc_str = "-" if r[5] is None else f"{r[5]:.3f}"
            print(f"{r[0]} | {r[1]:.6f} | {r[2]:.6f} | {r[3]:.6f} | {r[4]:.6f} | {acc_str} | {r[6]}")

        dir_overall = None
        yd = overall.get("yd", None)
        pd_all = overall.get("pd", None)
        if yd is not None and pd_all is not None and yd.numel() > 0 and pd_all.numel() > 0:
            try:
                L = min(yd.numel(), pd_all.numel())
                yd1 = yd[:L].float()
                pd1 = pd_all[:L]
                try:
                    if torch.isfinite(pd1).any() and (pd1.min() < 0 or pd1.max() > 1):
                        pd1 = torch.sigmoid(pd1)
                except Exception:
                    pd1 = torch.sigmoid(pd1)
                pd1 = torch.clamp(pd1, 0.0, 1.0)
                acc = ((pd1 >= 0.5).int() == yd1.int()).float().mean().item()
                brier = ((pd1 - yd1) ** 2).mean().item()
                auroc = None
                try:
                    from torchmetrics.classification import BinaryAUROC
                    au = BinaryAUROC()
                    auroc = float(au(pd1.detach().cpu(), yd1.detach().cpu()).item())
                except Exception:
                    auroc = None
                dir_overall = {"accuracy": acc, "brier": brier, "auroc": auroc}
                print("\nDirection (final):")
                msg = f"Accuracy: {acc:.3f} | Brier: {brier:.4f}"
                if auroc is not None:
                    msg += f" | AUROC: {auroc:.3f}"
                print(msg)
            except Exception as e:
                print(f"[WARN] Could not compute final direction metrics: {e}")

        try:
            import json
            out = {
                "decoded": True,
                "overall": {k: v for k, v in overall.items() if k in ("mae","rmse","mse","qlike")},
                "direction_overall": dir_overall,
                "per_asset": [
                    {"asset": r[0], "mae": r[1], "rmse": r[2], "mse": r[3], "qlike": r[4], "acc": r[5], "n": r[6]}
                    for r in rows
                ],
            }
            path = str(LOCAL_RUN_DIR / f"tft_val_asset_metrics_e{MAX_EPOCHS}_{RUN_SUFFIX}.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print(f"✓ Saved per-asset validation metrics (decoded, final) → {path}")
        except Exception as e:
            print(f"[WARN] Could not save final per-asset metrics: {e}")

        # Optionally save per-sample predictions for plotting
        try:
            import pandas as pd  # ensure pd is bound locally and not shadowed
            # Recompute decoded tensors from the stored device buffers
            g_cpu = torch.cat(self._g_dev).detach().cpu() if self._g_dev else None
            yv_cpu = torch.cat(self._yv_dev).detach().cpu() if self._yv_dev else None
            pv_cpu = torch.cat(self._pv_dev).detach().cpu() if self._pv_dev else None
            yd_cpu = torch.cat(self._yd_dev).detach().cpu() if self._yd_dev else None
            pd_cpu = torch.cat(self._pd_dev).detach().cpu() if self._pd_dev else None
            # decode vol back to physical scale
            if g_cpu is not None and yv_cpu is not None and pv_cpu is not None:
                yv_dec = self.vol_norm.decode(yv_cpu.unsqueeze(-1), group_ids=g_cpu.unsqueeze(-1)).squeeze(-1)
                pv_dec = self.vol_norm.decode(pv_cpu.unsqueeze(-1), group_ids=g_cpu.unsqueeze(-1)).squeeze(-1)
                # map group id -> name
                assets = [self.id_to_name.get(int(i), str(int(i))) for i in g_cpu.tolist()]
                # time index (may be missing)
                t_cpu = torch.cat(self._t_dev).detach().cpu() if self._t_dev else None
                df_out = pd.DataFrame({
                    "asset": assets,
                    "time_idx": t_cpu.numpy().tolist() if t_cpu is not None else [None] * len(assets),
                    "y_vol": yv_dec.numpy().tolist(),
                    "y_vol_pred": pv_dec.numpy().tolist(),
                })
                if yd_cpu is not None and pd_cpu is not None and yd_cpu.numel() > 0 and pd_cpu.numel() > 0:
                    # ensure pd is probability
                    pdp = pd_cpu
                    try:
                        if torch.isfinite(pdp).any() and (pdp.min() < 0 or pdp.max() > 1):
                            pdp = torch.sigmoid(pdp)
                    except Exception:
                        pdp = torch.sigmoid(pdp)
                    pdp = torch.clamp(pdp, 0.0, 1.0)
                    Lm = min(len(df_out), yd_cpu.numel(), pdp.numel())
                    df_out = df_out.iloc[:Lm].copy()
                    df_out["y_dir"] = yd_cpu[:Lm].numpy().tolist()
                    df_out["y_dir_prob"] = pdp[:Lm].numpy().tolist()
                # --- Attach real timestamps from a source DataFrame, if available ---
                try:
                    # Look for a likely source dataframe that contains ['asset','time_idx','Time']
                    candidate_names = ["raw_df", "raw_data", "full_df", "df", "val_df", "train_df", "test_df"]
                    src_df = None
                    for _name in candidate_names:
                        obj = globals().get(_name)
                        if isinstance(obj, pd.DataFrame) and {"asset", "time_idx", "Time"}.issubset(obj.columns):
                            src_df = obj[["asset", "time_idx", "Time"]].copy()
                            break
                    if src_df is not None:
                        # Harmonise dtypes prior to merge
                        src_df["asset"] = src_df["asset"].astype(str)
                        src_df["time_idx"] = pd.to_numeric(src_df["time_idx"], errors="coerce").astype("Int64").astype("int64")
                        df_out["asset"] = df_out["asset"].astype(str)
                        df_out["time_idx"] = pd.to_numeric(df_out["time_idx"], errors="coerce").astype("Int64").astype("int64")

                        # Merge on ['asset','time_idx'] to bring in the real Time column
                        df_out = df_out.merge(src_df, on=["asset", "time_idx"], how="left", validate="m:1")

                        # Coerce Time to timezone-naive pandas datetimes
                        df_out["Time"] = pd.to_datetime(df_out["Time"], errors="coerce")
                        try:
                            # If tz-aware, drop timezone info; if already naive this may raise — ignore
                            df_out["Time"] = df_out["Time"].dt.tz_localize(None)
                        except Exception:
                            pass
                    else:
                        print(
                            "[WARN] Could not locate a source dataframe with ['asset','time_idx','Time'] among candidates: "
                            "raw_df, raw_data, full_df, df (also checked val_df/train_df/test_df)."
                        )
                except Exception as e:
                    print(f"[WARN] Failed to attach real timestamps: {e}")
                pred_path = LOCAL_OUTPUT_DIR / f"tft_val_predictions_e{MAX_EPOCHS}_{RUN_SUFFIX}.parquet"
                df_out.to_parquet(pred_path, index=False)
                print(f"✓ Saved validation predictions (Parquet) → {pred_path}")
                try:
                    upload_file_to_gcs(str(pred_path), f"{GCS_OUTPUT_PREFIX}/{pred_path.name}")
                except Exception as e:
                    print(f"[WARN] Could not upload validation predictions: {e}")
        except Exception as e:
            print(f"[WARN] Could not save validation predictions: {e}")
            
# ---------------------------------------------------------------------
# Learnable Multi-Task Loss with Uncertainty-style Weights
# ---------------------------------------------------------------------
class LearnableMultiTaskLoss(nn.Module):
    """
    Two-loss combiner with learnable uncertainty-style weights.
    total = exp(-s1) * L1 + exp(-s2) * L2 + (s1 + s2)
    Initialize with desired starting weights w_i via s_i = -log(w_i).
    Expects y_pred to be a list/tuple: [pred_vol, pred_dir].
    """
    def __init__(self, loss_vol: nn.Module, loss_dir: nn.Module, init_weights=(1.0, 0.5)):
        super().__init__()
        self.loss_vol = loss_vol
        self.loss_dir = loss_dir
        w1, w2 = float(init_weights[0]), float(init_weights[1])
        # s_i = -log(w_i) so that exp(-s_i) = w_i at initialization
        s1 = -np.log(max(w1, 1e-8))
        s2 = -np.log(max(w2, 1e-8))
        self.s = nn.Parameter(torch.tensor([s1, s2], dtype=torch.float32))

    def forward(self, y_pred, target, **kwargs):
        # Unpack predictions
        if isinstance(y_pred, (list, tuple)) and len(y_pred) >= 2:
            p_vol, p_dir = y_pred[0], y_pred[1]
        else:
            raise TypeError("LearnableMultiTaskLoss expects y_pred as [vol_pred, dir_pred]")

        # Unpack targets -> expected shapes [..., pred_len] or [..., pred_len, 1]
        # Accept either a single tensor with last dim >=2 (multi-target) or a list
        if torch.is_tensor(target):
            t = target
            # Common PF shapes: [B, pred_len, n_targets] or [B, n_targets]
            if t.ndim == 3 and t.size(-1) >= 2:
                yv = t[..., 0]
                yd = t[..., 1]
            elif t.ndim == 2 and t.size(-1) >= 2:
                yv = t[:, 0]
                yd = t[:, 1]
            else:
                raise ValueError(f"Unexpected target shape for multi-task: {t.shape}")
        elif isinstance(target, (list, tuple)) and len(target) >= 2:
            yv, yd = target[0], target[1]
            if torch.is_tensor(yv) and yv.ndim >= 3 and yv.size(-1) == 1:
                yv = yv[..., 0]
            if torch.is_tensor(yd) and yd.ndim >= 3 and yd.size(-1) == 1:
                yd = yd[..., 0]
        else:
            raise TypeError("LearnableMultiTaskLoss expects target tensor with two channels or list of two tensors")

        L1 = self.loss_vol(p_vol, yv)
        L2 = self.loss_dir(p_dir, yd)
        if L1.ndim > 0: L1 = L1.mean()
        if L2.ndim > 0: L2 = L2.mean()

        weights = torch.exp(-self.s)  # [2]
        total = weights[0] * L1 + weights[1] * L2 + self.s.sum()
        return total

# -----------------------------------------------------------------------
# Mini-MLP dual head: one branch for volatility quantiles, one for direction logit
# -----------------------------------------------------------------------
class DualHead(nn.Module):
    def __init__(self, hidden_size: int, quantiles: int = 7):
        super().__init__()
        h = hidden_size // 2
        self.vol = nn.Sequential(
            nn.Linear(hidden_size, h),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(h, quantiles),
        )
        self.dir = nn.Sequential(
            nn.Linear(hidden_size, h),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(h, 1),  # single logit
        )

    def forward(self, x):
        vol_out = self.vol(x)
        dir_out = self.dir(x)
        return torch.cat([vol_out, dir_out], dim=-1)

# keep BCE pos_weight on the correct device
class StaticPosWeightBCE(nn.Module):
    def __init__(self, pos_weight: float, smoothing: float = 0.0):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.smoothing = float(smoothing)
    def forward(self, y_pred, target):
        target = target.float()
        if self.smoothing > 0:
            target = target * (1.0 - self.smoothing) + 0.5 * self.smoothing
        pw = torch.tensor(self.pos_weight, device=y_pred.device, dtype=y_pred.dtype)
        return F.binary_cross_entropy_with_logits(
            y_pred.squeeze(-1), target.squeeze(-1), pos_weight=pw
        )

class DualOutputModule(nn.Module):
    """Wrap two heads so output_layer(x) returns [vol_out, dir_out] for PF."""
    def __init__(self, vol_head: nn.Module, dir_head: nn.Module):
        super().__init__()
        self.vol = vol_head
        self.dir = dir_head
    def forward(self, x):
        return [self.vol(x), self.dir(x)]

# -----------------------------------------------------------------------
# Monkey-patch BaseModel.predict_step to support multi-target inference
# -----------------------------------------------------------------------

# Patch for TFT.py monkey-patch adjust
def monkey_patch_predict_step():
    from pytorch_forecasting.models.base._base_model import BaseModel
    import torch

    def _predict_step_multi(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """
        Custom predict_step that returns *all* target heads.
        Handles PF Output wrapper by extracting the underlying tensor.
        """
        # batch comes from TimeSeriesDataLoader: (x, y, weight)
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        with torch.no_grad():
            y_hat = self(x)

        # ――― Extract raw tensor(s) ―――
        pred = y_hat.prediction if hasattr(y_hat, "prediction") else y_hat

        # If PF returned a list (one tensor per target), stack them
        if isinstance(pred, (list, tuple)):
            processed = []
            for p in pred:                    # each p: [B,1,7] or [B,1,1,7]
                # squeeze singleton prediction‑length or target dims
                if p.ndim == 4 and p.shape[2] == 1:
                    p = p.squeeze(2)          # [B,1,7]
                if p.ndim == 3 and p.shape[1] == 1:
                    p = p.squeeze(1)          # [B,7]
                # ---- harmonise classification logits (size 1 or 2) -------------
                # if this target is a single logit (shape [...,1]), repeat to 7
                if p.ndim >= 2 and p.shape[-1] == 1:
                    p = p.repeat_interleave(7, dim=-1)      # [...,7]
                # If this target is a classification head with 2 logits,
                # convert to the probability of the positive class
                # and replicate it so the last dimension has length 7
                elif p.ndim >= 2 and p.shape[-1] == 2:
                    # convert logits -> prob of class 1
                    pos_prob = p.softmax(-1)[..., 1]          # shape [...,]
                    # make last dim explicit then repeat to length 7
                    pos_prob = pos_prob.unsqueeze(-1)         # [...,1]
                    p = pos_prob.repeat_interleave(7, dim=-1) # [...,7]
                processed.append(p)
            pred = torch.stack(processed, dim=1)  # [B,n_targets,7]

        # If tensor still has pred_len dim=1, squeeze it
        if torch.is_tensor(pred) and pred.ndim == 4 and pred.shape[2] == 1:
            pred = pred.squeeze(2)             # [B,n_targets,7]

        return {"prediction": pred, "x": x}

    BaseModel.predict_step = _predict_step_multi

# Activate the patch once at import time
monkey_patch_predict_step()

# -----------------------------------------------------------------------



# -----------------------------------------------------------------------
# Disable PF's plotting/logging hooks to avoid shape-related crashes
# during sanity check / validation. We still compute losses & metrics.
# -----------------------------------------------------------------------
try:
    from pytorch_forecasting.models.base._base_model import BaseModel
    import matplotlib.pyplot as plt

    def _noop_plot(self, *args, **kwargs):
        # return an empty Figure so callers expecting a Figure won't error
        fig = plt.figure()
        return fig

    def _noop_log_prediction(self, *args, **kwargs):
        return  # do nothing

    BaseModel.plot_prediction = _noop_plot
    BaseModel.log_prediction = _noop_log_prediction
    print("[INFO] Disabled PF plotting/logging for validation to avoid size-mismatch errors.")
except Exception as _e:
    print(f"[WARN] Could not disable PF plotting: {_e}")

# -----------------------------------------------------------------------
# Compute / device configuration (optimised for NVIDIA L4 on GCP)
# -----------------------------------------------------------------------
if torch.cuda.is_available():
    ACCELERATOR = "gpu"
    DEVICES = "auto"          # use all visible GPUs if more than one
    # default to bf16 but fall back to fp16 if unsupported (e.g., T4)
    PRECISION = "bf16-mixed"
    try:
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            PRECISION = "16-mixed"
    except Exception:
        try:
            major, _minor = torch.cuda.get_device_capability()
            if major < 8:  # pre-Ampere
                PRECISION = "16-mixed"
        except Exception:
            PRECISION = "16-mixed"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    ACCELERATOR = "mps"
    DEVICES = 1
    PRECISION = "16-mixed"
else:
    ACCELERATOR = "cpu"
    DEVICES = 1
    PRECISION = 32

# -----------------------------------------------------------------------
# CLI overrides for common CONFIG knobs
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# CLI overrides for common CONFIG knobs
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="TFT training with optional permutation importance", add_help=True)
parser.add_argument("--max_encoder_length", type=int, default=None, help="Max encoder length")
parser.add_argument("--max_epochs", type=int, default=None, help="Max training epochs")
parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
parser.add_argument("--early_stop_patience", "--patience", type=int, default=None, help="Early stopping patience")
parser.add_argument("--perm_len", type=int, default=None, help="Permutation block length for importance")
parser.add_argument("--perm_block_size", type=int, default=None, help="Alias for --perm_len (permutation block length)")
parser.add_argument(
    "--enable_perm_importance", "--enable-feature-importance",
    type=lambda s: str(s).lower() in ("1","true","t","yes","y","on"),
    default=None,
    help="Enable permutation feature importance (true/false)"
)
# Cloud paths / storage overrides
parser.add_argument("--gcs_bucket", type=str, default=None, help="GCS bucket name to read/write from")
parser.add_argument("--gcs_data_prefix", type=str, default=None, help="Full GCS prefix for data parquet folder")
parser.add_argument("--gcs_output_prefix", type=str, default=None, help="Full GCS prefix for outputs/checkpoints")
# Performance / input control
parser.add_argument("--data_dir", type=str, default=None, help="Local folder containing universal_*.parquet; if set, prefer local over GCS")
parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (defaults to CPU count - 1)")
parser.add_argument("--prefetch_factor", type=int, default=8, help="DataLoader prefetch factor (per worker)")
# Performance / input control
parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validate every N epochs")
parser.add_argument("--log_every_n_steps", type=int, default=200, help="How often to log train steps")
parser.add_argument("--learning_rate", type=float, default=None, help="Override model learning rate")
parser.add_argument("--resume", type=lambda s: str(s).lower() in ("1","true","t","yes","y","on"), default=True, help="Resume from latest checkpoint if available")
parser.add_argument("--fi_max_batches", type=int, default=20, help="Max val batches per feature in FI.")
# Parse known args so stray platform args do not crash the script
ARGS, _UNKNOWN = parser.parse_known_args()

# -----------------------------------------------------------------------
# CONFIG – tweak as required (GCS-aware)
# -----------------------------------------------------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET", "river-ml-bucket")
GCS_DATA_PREFIX = f"gs://{GCS_BUCKET}/Data/CleanedData"
GCS_OUTPUT_PREFIX = f"gs://{GCS_BUCKET}/Dissertation/Feature_Ablation"

# Apply CLI overrides (if provided)
if getattr(ARGS, "gcs_bucket", None):
    GCS_BUCKET = ARGS.gcs_bucket
    # recompute defaults if specific prefixes are not provided
    if not getattr(ARGS, "gcs_data_prefix", None):
        GCS_DATA_PREFIX = f"gs://{GCS_BUCKET}/CleanedData"
    if not getattr(ARGS, "gcs_output_prefix", None):
        GCS_OUTPUT_PREFIX = f"gs://{GCS_BUCKET}/Dissertation/Feature_Ablation"
if getattr(ARGS, "gcs_data_prefix", None):
    GCS_DATA_PREFIX = ARGS.gcs_data_prefix
if getattr(ARGS, "gcs_output_prefix", None):
    GCS_OUTPUT_PREFIX = ARGS.gcs_output_prefix

# Local ephemerals (good for GCE/Vertex)
LOCAL_DATA_DIR = Path(os.environ.get("LOCAL_DATA_DIR", "/tmp/data/CleanedData"))
LOCAL_OUTPUT_DIR = Path(os.environ.get("LOCAL_OUTPUT_DIR", "/tmp/feature_ablation"))
LOCAL_RUN_DIR = Path(os.environ.get("LOCAL_RUN_DIR", "/tmp/tft_run"))
LOCAL_LOG_DIR = LOCAL_RUN_DIR / "lightning_logs"
LOCAL_CKPT_DIR = LOCAL_RUN_DIR / "checkpoints"
for p in [LOCAL_DATA_DIR, LOCAL_OUTPUT_DIR, LOCAL_CKPT_DIR, LOCAL_LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------- TensorBoard Logger ----------------
logger = TensorBoardLogger(save_dir=str(LOCAL_LOG_DIR.parent), name=LOCAL_LOG_DIR.name)

# Remote checkpoint prefix on GCS
CKPT_GCS_PREFIX = f"{GCS_OUTPUT_PREFIX}/checkpoints"

# ---- Upload helper (GCS-aware) ----
def upload_file_to_gcs(local_path: str, gcs_uri: str):
    if fs is None:
        print(f"[WARN] GCS not available (gcsfs not installed); skipping upload: {gcs_uri}")
        return
    try:
        with fsspec.open(gcs_uri, "wb") as f_out, open(local_path, "rb") as f_in:
            shutil.copyfileobj(f_in, f_out)
        print(f"✓ Uploaded {local_path} → {gcs_uri}")
    except Exception as e:
        print(f"[WARN] Failed to upload {local_path} to {gcs_uri}: {e}")


# Prefer GCS if the files exist there
try:
    fs = fsspec.filesystem("gcs")
except Exception:
    fs = None  # gcsfs not installed / protocol unavailable

def gcs_exists(path: str) -> bool:
    if fs is None:
        return False
    try:
        return fs.exists(path)
    except Exception:
        return False

TRAIN_URI = f"{GCS_DATA_PREFIX}/universal_train.parquet"
VAL_URI   = f"{GCS_DATA_PREFIX}/universal_val.parquet"
TEST_URI  = f"{GCS_DATA_PREFIX}/universal_test.parquet"
READ_PATHS = [str(TRAIN_URI), str(VAL_URI), str(TEST_URI)]
'''
# If a local data folder is explicitly provided, use it and skip GCS
if getattr(ARGS, "data_dir", None):
    DATA_DIR = Path(ARGS.data_dir).expanduser().resolve()
    TRAIN_FILE = DATA_DIR / "universal_train.parquet"
    VAL_FILE   = DATA_DIR / "universal_val.parquet"
    TEST_FILE  = DATA_DIR / "universal_test.parquet"
    READ_PATHS = [str(TRAIN_FILE), str(VAL_FILE), str(TEST_FILE)]
elif not all(map(gcs_exists, [TRAIN_URI, VAL_URI, TEST_URI])):
    # Fallback to your original local paths
    DATA_DIR = Path("/Users/riverwest-gomila/Desktop/Data/CleanedData")
    TRAIN_FILE = DATA_DIR / "universal_train.parquet"
    VAL_FILE   = DATA_DIR / "universal_val.parquet"
    TEST_FILE  = DATA_DIR / "universal_test.parquet"
    READ_PATHS = [str(TRAIN_FILE), str(VAL_FILE), str(TEST_FILE)]
else:
    READ_PATHS = [str(TRAIN_URI), str(VAL_URI), str(TEST_URI)]
'''    

# --- Validate data availability early with helpful errors ---

def _is_gcs(p: str) -> bool:
    return str(p).startswith("gs://")

if all(_is_gcs(p) for p in READ_PATHS):
    if fs is None:
        raise RuntimeError(
            "Data paths point to GCS but 'gcsfs' is not installed. Install gcsfs or provide local files."
        )
else:
    missing_local = [p for p in READ_PATHS if not _is_gcs(p) and not Path(p).exists()]
    if missing_local:
        raise FileNotFoundError(
            f"Local data files not found: {missing_local}. Either upload your data to GCS (set GCS_BUCKET or pass --gcs_bucket) or ensure the local files exist."
        )

def get_resume_ckpt_path():
    # Prefer newest local checkpoint
    try:
        local_ckpts = sorted(LOCAL_CKPT_DIR.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if local_ckpts:
            return str(local_ckpts[0])
    except Exception:
        pass
    # Fallback: try GCS "last.ckpt" then the lexicographically latest .ckpt
    try:
        if fs is not None:
            last_uri = f"{CKPT_GCS_PREFIX}/last.ckpt"
            if fs.exists(last_uri):
                dst = LOCAL_CKPT_DIR / "last.ckpt"
                with fsspec.open(last_uri, "rb") as f_in, open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return str(dst)
            # else get the latest by name (filenames contain ISO timestamps)
            entries = fs.glob(f"{CKPT_GCS_PREFIX}/*.ckpt") or []
            if entries:
                latest = sorted(entries)[-1]
                dst = LOCAL_CKPT_DIR / Path(latest).name
                with fsspec.open(latest, "rb") as f_in, open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                return str(dst)
    except Exception as e:
        print(f"[WARN] Could not fetch resume checkpoint from GCS: {e}")
    return None

def mirror_local_ckpts_to_gcs():
    if fs is None:
        print("[WARN] GCS not available; skipping checkpoint upload.")
        return
    try:
        for p in LOCAL_CKPT_DIR.glob("*.ckpt"):
            remote = f"{CKPT_GCS_PREFIX}/{p.name}"
            with fsspec.open(remote, "wb") as f_out, open(p, "rb") as f_in:
                shutil.copyfileobj(f_in, f_out)
            print(f"✓ Mirrored checkpoint {p} → {remote}")
    except Exception as e:
        print(f"[WARN] Failed to mirror checkpoints: {e}")

class MirrorCheckpoints(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        mirror_local_ckpts_to_gcs()
    def on_exception(self, trainer, pl_module, err):
        mirror_local_ckpts_to_gcs()
    def on_train_end(self, trainer, pl_module):
        mirror_local_ckpts_to_gcs()

GROUP_ID: List[str] = ["asset"]
TIME_COL = "Time"
TARGETS  = ["realised_vol", "direction"]

MAX_ENCODER_LENGTH = 6
MAX_PRED_LENGTH    = 1

EMBEDDING_CARDINALITY = {}

BATCH_SIZE   = 2048
MAX_EPOCHS   = 5
EARLY_STOP_PATIENCE = 4
PERM_BLOCK_SIZE = 288

# Artifacts are written locally then uploaded to GCS
from datetime import datetime, timezone, timedelta
RUN_SUFFIX = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
MODEL_SAVE_PATH = (LOCAL_CKPT_DIR / f"tft_realised_vol_e{MAX_EPOCHS}_{RUN_SUFFIX}.ckpt")

SEED = 50
WEIGHT_DECAY = 0.00578350719515325     # weight decay for AdamW
GRADIENT_CLIP_VAL = 0.78    # gradient clipping value for Trainer
# Feature-importance controls
ENABLE_FEATURE_IMPORTANCE = True   # gate FI so you can toggle it
FI_MAX_BATCHES = 16       # number of val batches to sample for FI

# ---- Apply CLI overrides (only when provided) ----
if ARGS.batch_size is not None:
    BATCH_SIZE = int(ARGS.batch_size)
if ARGS.max_encoder_length is not None:
    MAX_ENCODER_LENGTH = int(ARGS.max_encoder_length)
if ARGS.max_epochs is not None:
    MAX_EPOCHS = int(ARGS.max_epochs)
if ARGS.early_stop_patience is not None:
    EARLY_STOP_PATIENCE = int(ARGS.early_stop_patience)
if ARGS.perm_len is not None:
    PERM_BLOCK_SIZE = int(ARGS.perm_len)
if ARGS.enable_perm_importance is not None:
    ENABLE_FEATURE_IMPORTANCE = bool(ARGS.enable_perm_importance)
if getattr(ARGS, "fi_max_batches", None) is not None:
    FI_MAX_BATCHES = int(ARGS.fi_max_batches)

# ---- Learning rate and resume CLI overrides ----
LR_OVERRIDE = float(ARGS.learning_rate) if getattr(ARGS, "learning_rate", None) is not None else None
RESUME_ENABLED = bool(getattr(ARGS, "resume", True))

# -----------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------
def load_split(path: str) -> pd.DataFrame:
    """
    Load a parquet split and guarantee we have usable `id` and timestamp columns.

    1. Converts `TIME_COL` to pandas datetime.
    2. Ensures an identifier column called `id` (or whatever `GROUP_ID[0]` is).
       If not present, auto-detects common synonyms and renames
    """
    path_str = str(path)
    df = pd.read_parquet(path)
    df = df.reset_index(drop=True).copy()

    # --- convert timestamp ---
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # --- identifier column handling ---
    if GROUP_ID[0] not in df.columns:
        cand = next(
            (c for c in df.columns if c.lower() in
             {"symbol", "ticker", "asset", "security_id", "instrument"}), None
        )
        if cand is None:
            raise ValueError(
                f"No identifier column '{GROUP_ID[0]}' in {path_str}. Edit GROUP_ID or rename your column."
            )

        df.rename(columns={cand: GROUP_ID[0]}, inplace=True)
        print(f"[INFO] Renamed '{cand}' ➜ '{GROUP_ID[0]}'")

    df[GROUP_ID[0]] = df[GROUP_ID[0]].astype(str)

    # --- target alias handling ------------------------------------------------
    TARGET_ALIASES = {
        "realised_vol": ["Realised_Vol", "rs_sigma", "realized_vol", "rv"],
        "direction":    ["Sign_Label", "sign_label", "Direction", "direction_label"],
    }
    for canonical, aliases in TARGET_ALIASES.items():
        if canonical not in df.columns:
            alias_found = next((a for a in aliases if a in df.columns), None)
            if alias_found:
                df.rename(columns={alias_found: canonical}, inplace=True)
                print(f"[INFO] Renamed '{alias_found}' ➜ '{canonical}'")
            else:
                # Only warn; the downstream code will raise if the column is truly required
                print(f"[WARN] Column '{canonical}' not found in {path_str} and no alias detected.")

    return df


def add_time_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Add monotonically increasing integer time index per asset."""
    df = df.sort_values(GROUP_ID + [TIME_COL])
    df["time_idx"] = (
        df.groupby(GROUP_ID)
          .cumcount()
          .astype("int64")
    )
    return df

# -----------------------------------------------------------------------
# Calendar features (help the model learn intraday/weekly seasonality)
# -----------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    minute_of_day = df[TIME_COL].dt.hour * 60 + df[TIME_COL].dt.minute
    df["sin_tod"] = np.sin(2 * np.pi * minute_of_day / 1440.0).astype("float32")
    df["cos_tod"] = np.cos(2 * np.pi * minute_of_day / 1440.0).astype("float32")
    dow = df[TIME_COL].dt.dayofweek
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0).astype("float32")
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0).astype("float32")
    df["Is_Weekend"] = (dow >= 5).astype("int8")
    return df


# -----------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print(
        f"[CONFIG] batch_size={BATCH_SIZE} | encoder={MAX_ENCODER_LENGTH} | epochs={MAX_EPOCHS} | "
        f"patience={EARLY_STOP_PATIENCE} | perm_len={PERM_BLOCK_SIZE} | "
        f"perm_importance={'on' if ENABLE_FEATURE_IMPORTANCE else 'off'} | fi_max_batches={FI_MAX_BATCHES}"
    )
    print("▶ Loading data …")
    train_df = add_time_idx(load_split(READ_PATHS[0]))
    val_df   = add_time_idx(load_split(READ_PATHS[1]))
    test_df  = add_time_idx(load_split(READ_PATHS[2]))
    # -------------------------------------------------------------------
    # Compute per‑asset median realised_vol scale (rv_scale) **once** on the TRAIN split
    # and attach it to every split. This is used only as a **fallback** for manual decode
    # if a normaliser decode is not available.
    # -------------------------------------------------------------------
    asset_scales = (
        train_df.groupby("asset", observed=True)["realised_vol"]
                .median()
                .clip(lower=1e-8)                 # guard against zeros
                .rename("rv_scale")
                .reset_index()
    )

    # Attach rv_scale without using the deprecated/invalid `inplace` kwarg
    asset_scale_map = asset_scales.set_index("asset")["rv_scale"]
    for df in (train_df, val_df, test_df):
        # map() preserves the original row order and keeps dtype float64
        df["rv_scale"] = df["asset"].map(asset_scale_map)
        # If an asset appears only in val/test, fall back to overall median
        df["rv_scale"].fillna(asset_scale_map.median(), inplace=True)

    # Add calendar features to all splits
    train_df = add_calendar_features(train_df)
    val_df   = add_calendar_features(val_df)
    test_df  = add_calendar_features(test_df)


    # -----------------------------------------------------------------------
    # Feature definitions
    # -----------------------------------------------------------------------
    static_categoricals = GROUP_ID
    static_reals: List[str] = []

    base_exclude = set(GROUP_ID + [TIME_COL, "time_idx", "rv_scale"] + TARGETS)

    all_numeric = [c for c, dt in train_df.dtypes.items()
                   if (c not in base_exclude) and pd.api.types.is_numeric_dtype(dt)]

    # Specify future-known and unknown real features
    calendar_cols = ["sin_tod", "cos_tod", "sin_dow", "cos_dow"]
    time_varying_known_reals = calendar_cols + ["Is_Weekend"]
    time_varying_unknown_reals = [c for c in all_numeric if c not in (calendar_cols + ["Is_Weekend"]) ]



    # -----------------------------------------------------------------------
    # TimeSeriesDataSets
    # -----------------------------------------------------------------------
    def build_dataset(df: pd.DataFrame, predict: bool) -> TimeSeriesDataSet:
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=TARGETS,
            group_ids=GROUP_ID,
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_prediction_length=MAX_PRED_LENGTH,
            # ------------------------------------------------------------------
            # Target normalisation
            #   • realised_vol → GroupNormalizer(asinh, per‑asset scaling)
            #   • direction    → identity (classification logits)
            # ------------------------------------------------------------------
            target_normalizer = MultiNormalizer([
                GroupNormalizer(
                    groups=GROUP_ID,
                    center=False,
                    scale_by_group=True,
                    transformation="asinh",
                ),
                TorchNormalizer(method="identity", center=False),   # direction
            ]),
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,   # known at prediction time
            time_varying_unknown_reals=time_varying_unknown_reals # same set, allows learning lagged targets
            ,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
            predict_mode=predict
        )


    print("▶ Building TimeSeriesDataSets …")
    training_dataset = build_dataset(train_df, predict=False)
    validation_dataset = build_dataset(val_df, predict=False)
    # use predict=False so we obtain one sample **per time‑step**, not just the last step of each series
    test_dataset = build_dataset(test_df, predict=False)

    batch_size = min(BATCH_SIZE, len(training_dataset))

    # DataLoader performance knobs
    default_workers = max(2, (os.cpu_count() or 4) - 1)
    worker_cnt = int(ARGS.num_workers) if getattr(ARGS, "num_workers", None) is not None else default_workers
    prefetch = int(getattr(ARGS, "prefetch_factor", 8))
    pin = torch.cuda.is_available()
    use_persist = worker_cnt > 0

    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=worker_cnt,
        pin_memory=pin,
        persistent_workers=use_persist,
        prefetch_factor=prefetch,
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=worker_cnt,
        persistent_workers=use_persist,
        prefetch_factor=prefetch,
        pin_memory=pin,
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=worker_cnt,
        persistent_workers=use_persist,
        prefetch_factor=prefetch,
        pin_memory=pin,
    )

    # ---- derive id→asset-name mapping for callbacks ----
    asset_vocab = (
        training_dataset.get_parameters()["categorical_encoders"]["asset"].classes_
    )
    rev_asset = {i: lbl for i, lbl in enumerate(asset_vocab)}


    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    seed_everything(SEED, workers=True)
    # Loss and output_size for multi-target: realised_vol (quantile regression), direction (classification)
    print("▶ Building model …")
    print(f"[LR] learning_rate={LR_OVERRIDE if LR_OVERRIDE is not None else 0.00187}")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=64,
        attention_head_size=2,
        dropout=0.0833704625250354,
        hidden_continuous_size=16,
        learning_rate=(LR_OVERRIDE if LR_OVERRIDE is not None else 0.0019),
        optimizer="AdamW",
        optimizer_params={"weight_decay": WEIGHT_DECAY},
        output_size=[7, 1],  # 7 quantiles + 1 logit
        loss=MultiLoss([
            AsymmetricQuantileLoss(
                quantiles=[0.05, 0.165, 0.25, 0.5, 0.75, 0.835, 0.95],
                underestimation_factor=1.115,
            ),
            LabelSmoothedBCE(smoothing=0.1),
        ], weights=[1.0, 0.1]),
        logging_metrics=[],
        log_interval=50,
        log_val_interval=10,
        reduce_on_plateau_patience=5,
        reduce_on_plateau_min_lr=1e-5,
    )

    # --- Swap in LearnableMultiTaskLoss and DualHead (patched) ---
    try:
        _LossClass = AsymmetricQuantileLoss  # use your existing AQL if present
    except NameError:
        from pytorch_forecasting.metrics import QuantileLoss
        class AsymmetricQuantileLoss(QuantileLoss):
            def __init__(self, quantiles, underestimation_init: float = 1.115, learnable_under: bool = True, reg_lambda: float = 1e-3, **kwargs):
                super().__init__(quantiles=quantiles, **kwargs)
                self.learnable_under = bool(learnable_under)
                self.reg_lambda = float(reg_lambda)
                a0 = max(float(underestimation_init), 1.0)
                u0 = np.log(np.expm1(a0 - 1.0) + 1e-12)
                param = torch.tensor(u0, dtype=torch.float32)
                if self.learnable_under:
                    self.u = nn.Parameter(param)
                else:
                    self.register_buffer("u", param)
            def _alpha(self): 
                return 1.0 + F.softplus(self.u)
            def loss_per_prediction(self, y_pred, target):
                diff = target.unsqueeze(-1) - y_pred
                q = torch.tensor(self.quantiles, device=y_pred.device, dtype=y_pred.dtype).view(*([1] * (diff.ndim - 1)), -1)
                alpha = self._alpha().to(y_pred.device, dtype=y_pred.dtype)
                return torch.where(diff >= 0, alpha * q * diff, (1 - q) * (-diff))
            def forward(self, y_pred, target, **kwargs):
                base = super().forward(y_pred, target, **kwargs)
                if self.reg_lambda > 0:
                    base = base + self.reg_lambda * (self._alpha() - 1.0) ** 2
                return base
        _LossClass = AsymmetricQuantileLoss

    # Loss: underestimation starts at 1.115; direction pos_weight=1.1
    # Build vol loss compatible with either AQL variant
    try:
        vol_loss = _LossClass(
            quantiles=[0.05, 0.165, 0.25, 0.5, 0.75, 0.835, 0.95],
            underestimation_init=1.115,
            learnable_under=True,
            reg_lambda=1e-3,
        )
    except (TypeError, ValueError):
        # Older in-file AQL expects `underestimation_factor`
        vol_loss = _LossClass(
            quantiles=[0.05, 0.165, 0.25, 0.5, 0.75, 0.835, 0.95],
            underestimation_factor=1.115,
        )

    tft.loss = LearnableMultiTaskLoss(
        loss_vol=vol_loss,
        loss_dir=StaticPosWeightBCE(pos_weight=1.1),
        init_weights=(1.0, 0.5),
    )

    # Replace default linear heads with DualHead and bias-init direction from pos_weight=1.1
    dual_head = DualHead(hidden_size=int(getattr(tft.hparams, "hidden_size", 64)))
    p0 = 1.0 / (1.0 + 1.1)
    bias0 = float(np.log(p0 / (1.0 - p0)))
    with torch.no_grad():
        dual_head.dir[-1].bias.data.fill_(bias0)
    tft.output_layer = DualOutputModule(dual_head.vol, dual_head.dir)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    # ----------------- Permutation importance loss scaling helper -----------------
    def _scale_loss(val):
        try:
            return (val - 0.3) * 10.0
        except Exception:
            return val

    class ComponentLossLogger(pl.Callback):
      def __init__(self, path: str = str(LOCAL_RUN_DIR / f"tft_component_losses_e{MAX_EPOCHS}_{RUN_SUFFIX}.csv")):
          super().__init__()
          self.path = path
          self.rows = []
      def _row(self, trainer, pl_module, phase: str):
          try:
              loss_mod = getattr(pl_module, "loss", None)
              L1 = float(loss_mod.last_L1.cpu().item()) if hasattr(loss_mod, "last_L1") else float('nan')
              L2 = float(loss_mod.last_L2.cpu().item()) if hasattr(loss_mod, "last_L2") else float('nan')
              s = loss_mod.last_s.cpu().tolist() if hasattr(loss_mod, "last_s") else [None, None]
              w = loss_mod.last_w.cpu().tolist() if hasattr(loss_mod, "last_w") else [None, None]
              r = {
                  "epoch": trainer.current_epoch,
                  "phase": phase,
                  "L1": L1,
                  "L2": L2,
                  "s1": s[0] if s else None,
                  "s2": s[1] if s and len(s) > 1 else None,
                  "w1": w[0] if w else None,
                  "w2": w[1] if w and len(w) > 1 else None,
              }
              # quick TB logs if available
              try:
                  if L1 is not None:
                      trainer.logger.experiment.add_scalar(f"components/{phase}_L1_vol", L1, trainer.global_step)
                  if L2 is not None:
                      trainer.logger.experiment.add_scalar(f"components/{phase}_L2_dir", L2, trainer.global_step)
              except Exception:
                  pass
              return r
          except Exception:
              return None
      def on_train_epoch_end(self, trainer, pl_module):
          r = self._row(trainer, pl_module, "train")
          if r:
              self.rows.append(r)
              try:
                  ep = int(getattr(trainer, "current_epoch", -1)) + 1
                  l1 = r.get("L1")
                  l2 = r.get("L2")
                  w1 = r.get("w1")
                  w2 = r.get("w2")
                  msg = f"[TRAIN EPOCH {ep}] L1_vol={l1:.6f} L2_dir={l2:.6f}"
                  if w1 is not None and w2 is not None:
                      msg += f" | w=({w1:.3f},{w2:.3f})"
                  print(msg)
              except Exception:
                  pass
      def on_validation_epoch_end(self, trainer, pl_module):
          r = self._row(trainer, pl_module, "val")
          if r:
              self.rows.append(r)
              try:
                  ep = int(getattr(trainer, "current_epoch", -1)) + 1
                  l1 = r.get("L1")
                  l2 = r.get("L2")
                  w1 = r.get("w1")
                  w2 = r.get("w2")
                  msg = f"[VAL   EPOCH {ep}] L1_vol={l1:.6f} L2_dir={l2:.6f}"
                  if w1 is not None and w2 is not None:
                      msg += f" | w=({w1:.3f},{w2:.3f})"
                  print(msg)
              except Exception:
                  pass
      def on_train_end(self, trainer, pl_module):
          try:
              import pandas as pd
              if self.rows:
                  pd.DataFrame(self.rows).to_csv(self.path, index=False)
                  print(f"✓ Saved component losses → {self.path}")
                  try:
                      upload_file_to_gcs(self.path, f"{GCS_OUTPUT_PREFIX}/{os.path.basename(self.path)}")
                  except Exception as e:
                      print(f"[WARN] Could not upload component losses: {e}")
          except Exception as e:
              print(f"[WARN] Could not save component losses: {e}")

    class LossHistory(pl.Callback):
      def __init__(self, path: str = str(LOCAL_RUN_DIR / f"tft_loss_history_e{MAX_EPOCHS}_{RUN_SUFFIX}.csv")):
          self.path = path
          self.records = []
      def on_train_epoch_end(self, trainer, pl_module):
          loss = trainer.callback_metrics.get("train_loss_epoch")
          if loss is not None:
              self.records.append(
                  {"epoch": int(getattr(trainer, "current_epoch", -1)), "train_loss": float(f"{float(loss):.8f}")}
              )
      def on_train_end(self, trainer, pl_module):
          if self.records:
              import pandas as pd
              pd.DataFrame(self.records).to_csv(self.path, index=False)
              print(f"✓ Saved loss history → {self.path}")
              try:
                  upload_file_to_gcs(self.path, f"{GCS_OUTPUT_PREFIX}/{os.path.basename(self.path)}")
              except Exception as e:
                  print(f"[WARN] Could not upload loss history: {e}")


    class HighPrecisionTQDM(TQDMProgressBar):
        """Progress bar that shows key metrics with higher precision."""
        def get_metrics(self, trainer, pl_module):
            metrics = super().get_metrics(trainer, pl_module)
            # Pull raw values from callback_metrics when possible and format to 8 dp
            keys = (
                "loss", "train_loss_step", "train_loss", "train_loss_epoch",
                "val_loss", "val_loss_decoded", "val_mae_dec", "val_rmse_dec"
            )
            for k in keys:
                try:
                    if k in trainer.callback_metrics:
                        v = float(trainer.callback_metrics[k])
                        metrics[k] = f"{v:.8f}"
                    elif k in metrics:
                        # fallback to whatever Lightning provided
                        v = float(metrics[k])
                        metrics[k] = f"{v:.8f}"
                except Exception:
                    # ignore non-float or missing
                    pass
            return metrics

    class StepLossSaver(pl.Callback):
        """Saves train_loss_step every logged batch with high precision."""
        def __init__(self, path: str = str(LOCAL_RUN_DIR / f"tft_step_losses_e{MAX_EPOCHS}_{RUN_SUFFIX}.csv")):
            self.path = path
            self.rows = []
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            try:
                v = trainer.callback_metrics.get("train_loss_step")
                if v is not None:
                    self.rows.append({
                        "global_step": int(getattr(trainer, "global_step", 0)),
                        "epoch": int(getattr(trainer, "current_epoch", -1)),
                        "train_loss_step": float(f"{float(v):.8f}")
                    })
            except Exception:
                pass
        def on_train_end(self, trainer, pl_module):
            if not self.rows:
                return
            try:
                import pandas as pd
                pd.DataFrame(self.rows).to_csv(self.path, index=False)
                print(f"✓ Saved step loss history → {self.path}")
                try:
                    upload_file_to_gcs(self.path, f"{GCS_OUTPUT_PREFIX}/{os.path.basename(self.path)}")
                except Exception as e:
                    print(f"[WARN] Could not upload step loss history: {e}")
            except Exception as e:
                print(f"[WARN] Could not save step loss history: {e}")

    class EpochPrinter(pl.Callback):
        """Print lightweight, periodic training status updates per epoch.
        Shows epoch number, train/val loss (if available), and current LR.
        """
        def __init__(self, total_epochs: int):
            super().__init__()
            self.total = int(total_epochs)
        def _format_lr(self, trainer):
            try:
                opt = trainer.optimizers[0]
                lr = opt.param_groups[0].get("lr", None)
                return None if lr is None else f"{float(lr):.2e}"
            except Exception:
                return None
        def on_train_epoch_start(self, trainer, pl_module):
            # +1 to present human-friendly epoch numbering
            print(f"▶ Epoch {trainer.current_epoch + 1}/{self.total} — training …")
        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            train_loss = metrics.get("train_loss_epoch")
            val_loss = metrics.get("val_loss")
            val_mae_dec = metrics.get("val_mae_dec")
            val_rmse_dec = metrics.get("val_rmse_dec")
            val_loss_decoded = metrics.get("val_loss_decoded")
            lr_str = self._format_lr(trainer)
            parts = [f"✓ Epoch {trainer.current_epoch + 1}/{self.total} done"]
            if train_loss is not None:
                try:
                    parts.append(f"train_loss={float(train_loss):.8f}")
                except Exception:
                    pass
            if val_loss is not None:
                try:
                    parts.append(f"val_loss={float(val_loss):.8f}")
                except Exception:
                    pass
            if val_mae_dec is not None:
                try:
                    parts.append(f"val_mae_dec={float(val_mae_dec):.8f}")
                except Exception:
                    pass
            if val_rmse_dec is not None:
                try:
                    parts.append(f"val_rmse_dec={float(val_rmse_dec):.8f}")
                except Exception:
                    pass
            if val_loss_decoded is not None:
                try:
                    parts.append(f"val_loss_decoded={float(val_loss_decoded):.8f}")
                except Exception:
                    pass
            if lr_str is not None:
                parts.append(f"lr={lr_str}")
            print(" | ".join(parts))
        def on_validation_end(self, trainer, pl_module):
            # Print the latest best checkpoint path if available
            try:
                # `best_ckpt_cb` is defined in the same module where callbacks are assembled
                from builtins import best_ckpt_cb  # noqa: F401 (hint to linters; safe at runtime)
            except Exception:
                pass
            try:
                ckpt_cb = None
                for cb in trainer.callbacks:
                    if isinstance(cb, pl.callbacks.ModelCheckpoint):
                        ckpt_cb = cb
                        break
                if ckpt_cb and getattr(ckpt_cb, "best_model_path", ""):
                    print(f"↳ Best so far: {ckpt_cb.best_model_path}")
            except Exception:
                pass

    best_ckpt_cb = ModelCheckpoint(
        monitor="val_loss_decoded",
        mode="min",
        save_top_k=1,
        save_last=True,  # writes last.ckpt
        filename=f"tft_best_e{MAX_EPOCHS}_{RUN_SUFFIX}",
        dirpath=str(LOCAL_CKPT_DIR),
    )
    ckpt_uploader_cb = MirrorCheckpoints()

    callbacks = [
        HighPrecisionTQDM(),
        best_ckpt_cb,
        ckpt_uploader_cb,
        EpochPrinter(MAX_EPOCHS),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss_decoded", patience=EARLY_STOP_PATIENCE, mode="min"),
        ComponentLossLogger(),
        LossHistory(),
        StepLossSaver(),
        PerAssetMetrics(
            rev_asset,
            vol_normalizer=(
                training_dataset.target_normalizer.normalizers[0]
                if hasattr(training_dataset.target_normalizer, "normalizers")
                else training_dataset.target_normalizer
            ),
        ),
    ]
    print("▶ Creating Trainer …")
trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator=ACCELERATOR,
    devices=DEVICES,
    precision=PRECISION,
    default_root_dir=str(LOCAL_RUN_DIR),
    callbacks=callbacks,
    logger=logger,
    check_val_every_n_epoch=int(getattr(ARGS, "check_val_every_n_epoch", 1)),
    log_every_n_steps=int(getattr(ARGS, "log_every_n_steps", 200)),
    gradient_clip_val=GRADIENT_CLIP_VAL,
)

# Resume logic with epoch bump if needed
resume_ckpt = get_resume_ckpt_path() if RESUME_ENABLED else None
if resume_ckpt:
    print(f"▶ Resuming from checkpoint: {resume_ckpt}")
    try:
        import torch as _torch
        _meta = _torch.load(resume_ckpt, map_location="cpu")
        _epoch = None
        for k in ("epoch", "current_epoch"):
            if isinstance(_meta, dict) and k in _meta:
                _epoch = int(_meta[k])
                break
        if _epoch is not None:
            need = int(_epoch) + 1
            fit_loop = getattr(trainer, "fit_loop", None)
            if fit_loop is not None and getattr(fit_loop, "max_epochs", None) is not None:
                if fit_loop.max_epochs < need:
                    print(f"[INFO] Bumping Trainer.max_epochs {fit_loop.max_epochs} → {need} to match checkpoint epoch {_epoch}.")
                    fit_loop.max_epochs = need
    except Exception as e:
        print(f"[WARN] Could not inspect checkpoint epoch: {e}. Proceeding without bump.")

print("▶ Starting training …")
trainer.fit(tft, train_dataloader, val_dataloader, ckpt_path=resume_ckpt)
# -----------------------------------------------------------------------
# Permutation Feature Importance (fast, limited val batches)

if ENABLE_FEATURE_IMPORTANCE:
    try:
        print("▶ Running permutation feature importance …")

        rng = np.random.default_rng(SEED)

        def _permute_by_blocks(df_in: pd.DataFrame, col: str, group_col: str = GROUP_ID[0], block_size: int = PERM_BLOCK_SIZE) -> pd.DataFrame:
            """Shuffle contiguous blocks of length `block_size` within each group for column `col`.
            Keeps distribution roughly intact while breaking temporal alignment.
            """
            df = df_in.copy()
            if col not in df.columns:
                return df
            for g, sub in df.groupby(group_col, sort=False):
                idx = sub.index.to_numpy()
                if idx.size == 0:
                    continue
                # Build block index array: [0,0,...,1,1,...,2,2,...]
                n_blocks = max(1, int(np.ceil(idx.size / float(block_size))))
                block_ids = np.repeat(np.arange(n_blocks), block_size)[: idx.size]
                # Permute block order
                rng.shuffle(block_ids)
                # Reorder indices by permuted block ids but keep within-block order
                order = np.argsort(block_ids, kind="stable")
                df.loc[idx, col] = sub[col].to_numpy()[order]
            return df

        # Build a lightweight validation-only Trainer with a cap on batches
        fi_trainer = Trainer(
            accelerator=ACCELERATOR,
            devices=DEVICES,
            precision=PRECISION,
            logger=False,
            enable_checkpointing=False,
            limit_val_batches=int(FI_MAX_BATCHES),
            enable_progress_bar=False,
        )
        # Reload best checkpoint before FI for consistent evaluation
        try:
            best_ckpt_path = best_ckpt_cb.best_model_path
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                print(f"[FI] Loading best model checkpoint: {best_ckpt_path}")
                tft = tft.__class__.load_from_checkpoint(best_ckpt_path)

                # Re-attach DualHead after loading checkpoint (keep loss etc. unchanged)
                dual_head = DualHead(hidden_size=int(getattr(tft.hparams, "hidden_size", 64)))
                p0 = 1.0 / (1.0 + 1.1)                      # pos_weight=1.1 prior
                bias0 = float(np.log(p0 / (1.0 - p0)))
                with torch.no_grad():
                    dual_head.dir[-1].bias.data.fill_(bias0)
                tft.output_layer = DualOutputModule(dual_head.vol, dual_head.dir)
            else:
                print("[FI][WARN] No best checkpoint found; using current model state.")
        except Exception as e:
            print(f"[FI][WARN] Failed to load best checkpoint: {e}")

        # Baseline loss on the *original* validation set (limited batches)
        base_metrics = fi_trainer.validate(tft, dataloaders=val_dataloader, verbose=False)
        baseline = float(base_metrics[0].get("val_loss", np.nan)) if base_metrics else np.nan
        print(f"[FI] Baseline val_loss (first {FI_MAX_BATCHES} batches): {baseline:.8f}")

        # Candidate features: both unknown and known reals used by the model
        fi_features = list(dict.fromkeys(time_varying_unknown_reals + time_varying_known_reals))

        # Use single-threaded DataLoader for FI to avoid spawning many worker pools
        fi_num_workers = 0
        fi_persist = False
        fi_pin = False
        loader_kwargs = dict(
            train=False,
            batch_size=batch_size,
            num_workers=fi_num_workers,
            persistent_workers=fi_persist,
            pin_memory=fi_pin,
        )

        # Checkpoint file to allow resuming FI if interrupted (stable name)
        fi_ckpt_path = LOCAL_OUTPUT_DIR / "tft_perm_importance_checkpoint.csv"
        done_features = set()
        if fi_ckpt_path.exists():
            try:
                import pandas as _pd
                _ck = _pd.read_csv(fi_ckpt_path)
                if "feature" in _ck.columns:
                    done_features = set(_ck["feature"].astype(str).tolist())
                    print(f"[FI] Resume detected: {len(done_features)} features already computed, will skip them.")
            except Exception as _e:
                print(f"[FI][WARN] Could not read FI checkpoint: {_e}")

        fi_rows = []
        for feat in fi_features:
            if feat in done_features:
                # print(f"[FI] {feat}: skipped (already in checkpoint)")
                continue
            try:
                perm_df = _permute_by_blocks(val_df, feat)
                perm_ds = build_dataset(perm_df, predict=False)
                perm_loader = perm_ds.to_dataloader(**loader_kwargs)
                m = fi_trainer.validate(tft, dataloaders=perm_loader, verbose=False)
                perm_loss = float(m[0].get("val_loss", np.nan)) if m else np.nan
                delta = perm_loss - baseline if (np.isfinite(perm_loss) and np.isfinite(baseline)) else np.nan
                print(f"  • {feat:>30s} Δloss = {delta:+.6f}")
                fi_rows.append({"feature": feat, "baseline_val_loss": baseline, "perm_val_loss": perm_loss, "delta": delta})
                # Append the row to the checkpoint file
                try:
                    import pandas as _pd
                    _row_df = _pd.DataFrame([fi_rows[-1]])
                    _row_df.to_csv(fi_ckpt_path, mode="a", header=not fi_ckpt_path.exists(), index=False)
                except Exception as _e:
                    print(f"[FI][WARN] Could not append FI checkpoint: {_e}")
                try:
                    del perm_loader
                    del perm_ds
                except Exception:
                    pass
                import gc as _gc
                _gc.collect()
            except Exception as e:
                print(f"[FI][WARN] Failed on feature '{feat}': {e}")

        import gc as _gc
        _gc.collect()

        # Save to CSV locally and upload to GCS (merge checkpoint + current rows)
        try:
            import pandas as _pd
            _all_rows = fi_rows.copy()
            if fi_ckpt_path.exists():
                try:
                    _prev = _pd.read_csv(fi_ckpt_path)
                    _all_rows.extend(_prev.to_dict("records"))
                except Exception as _e:
                    print(f"[FI][WARN] Could not merge FI checkpoint: {_e}")
            # Drop duplicates by feature, keep last (current run wins)
            _df = _pd.DataFrame(_all_rows)
            if not _df.empty:
                _df = _df.drop_duplicates(subset=["feature"], keep="last")
                fi_path = LOCAL_OUTPUT_DIR / f"tft_perm_importance_e{MAX_EPOCHS}_{RUN_SUFFIX}.csv"
                _df.sort_values("delta", ascending=False).to_csv(fi_path, index=False)
                print(f"✓ Saved permutation importance → {fi_path}")
                try:
                    upload_file_to_gcs(str(fi_path), f"{GCS_OUTPUT_PREFIX}/{fi_path.name}")
                except Exception as ue:
                    print(f"[WARN] Could not upload FI CSV: {ue}")
            else:
                print("[FI][WARN] No FI rows collected; nothing to save.")
        except Exception as se:
            print(f"[FI][WARN] Could not save FI CSV: {se}")

    except Exception as e:
        print(f"[FI][WARN] Permutation importance failed: {e}")
else:
    print("[FI] Skipped (ENABLE_FEATURE_IMPORTANCE is False).")