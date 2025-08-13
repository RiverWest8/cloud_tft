

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
import pandas as _pd
pd = _pd  # Ensure pd always refers to pandas module
import lightning as pl
from lightning.pytorch import Trainer, seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
def _permute_series_inplace(df: pd.DataFrame, col: str, block: int, group_col: str = "asset") -> None:
    if col not in df.columns:
        return
    if group_col not in df.columns:
        vals = df[col].values.copy()
        np.random.shuffle(vals)
        df[col] = vals
        return
    for _, idx in df.groupby(group_col, observed=True).groups.items():
        idx = np.asarray(list(idx))
        if block and block > 1:
            shift = np.random.randint(1, max(2, len(idx)))
            df.loc[idx, col] = df.loc[idx, col].values.take(np.arange(len(idx)) - shift, mode='wrap')
        else:
            vals = df.loc[idx, col].values.copy()
            np.random.shuffle(vals)
            df.loc[idx, col] = vals

# ------------------ Added imports for new FI block ------------------

from pytorch_forecasting import (
    BaseModel,
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

import pytorch_forecasting as pf
import inspect

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
        diff = target.unsqueeze(-1) - y_pred  # positive ⇒ under‑prediction
        q = torch.tensor(self.quantiles, device=y_pred.device).view(
            *([1] * (diff.ndim - 1)), -1
        )
        alpha = torch.as_tensor(self.underestimation_factor, dtype=y_pred.dtype, device=y_pred.device)
        # Heavier penalty when under‑predicting (diff >= 0)
        loss = torch.where(
            diff >= 0,
            alpha * q * diff,
            (1 - q) * (-diff),
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
        pdir = torch.cat(self._pd_dev).to(device) if self._pd_dev else None

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
        pd_cpu = pdir.detach().cpu() if pdir is not None else None

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

        # —— expose decoded metrics and define a combined val_loss = MAE + RMSE + 0.05 * DirBCE ——
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
                    dir_bce_t = F.binary_cross_entropy_with_logits(pt, yt_s)
                else:
                    p = _torch.clamp(pt, 1e-7, 1 - 1e-7)
                    yt_s = yt * 0.9 + 0.05
                    dir_bce_t = F.binary_cross_entropy(p, yt_s)
                dir_bce = float(dir_bce_t.item())

            # Base composite (decoded): MAE + RMSE
            comp_val = float(overall_mae) + float(overall_rmse)
            # Add small weight on direction (BCE) if available
            if dir_bce is not None and np.isfinite(dir_bce):
                comp_val_full = comp_val + 0.05 * float(dir_bce)
            else:
                comp_val_full = comp_val

            # Log individual components and the combined loss that we want to track
            trainer.callback_metrics["val_mae_dec"] = _torch.tensor(float(overall_mae))
            trainer.callback_metrics["val_rmse_dec"] = _torch.tensor(float(overall_rmse))
            trainer.callback_metrics["val_dir_bce"] = (
                _torch.tensor(float(dir_bce)) if dir_bce is not None else _torch.tensor(float('nan'))
            )
            # The key below is what Lightning callbacks (EarlyStopping/Checkpoint) commonly monitor by default
            trainer.callback_metrics["val_loss"] = _torch.tensor(comp_val_full)
            # Keep a named alias for clarity in logs
            trainer.callback_metrics["val_loss_decoded"] = _torch.tensor(comp_val_full)
        except Exception:
            pass

        self._last_rows = sorted(rows, key=lambda r: r[-1], reverse=True)
        self._last_overall = {
            "mae": overall_mae,
            "rmse": overall_rmse,
            "mse": overall_mse,
            "qlike": overall_qlike,
            "val_loss": float(overall_mae + overall_rmse) + (
                0.05 * float(trainer.callback_metrics.get("val_dir_bce", torch.tensor(float('nan'))).item())
                if torch.is_tensor(trainer.callback_metrics.get("val_dir_bce")) and torch.isfinite(trainer.callback_metrics["val_dir_bce"]) else 0.0
            ),
            "dir_bce": float(trainer.callback_metrics.get("val_dir_bce", torch.tensor(float('nan'))).item())
                if torch.is_tensor(trainer.callback_metrics.get("val_dir_bce")) else None,
            "yd": yd_cpu,
            "pd": pd_cpu,
        }
        

        # ---- concise per-epoch validation metrics printout (overall only) ----
        try:
            epoch_num = int(getattr(trainer, "current_epoch", -1)) + 1
        except Exception:
            epoch_num = None
        try:
            # Compute overall accuracy if direction available
            acc = None
            if yd_cpu is not None and pd_cpu is not None and yd_cpu.numel() > 0 and pd_cpu.numel() > 0:
                yt = yd_cpu.float()
                pt = pd_cpu
                try:
                    if torch.isfinite(pt).any() and (pt.min() < 0 or pt.max() > 1):
                        pt = torch.sigmoid(pt)
                except Exception:
                    pt = torch.sigmoid(pt)
                pt = torch.clamp(pt, 0.0, 1.0)
                acc = ((pt >= 0.5).int() == yt.int()).float().mean().item()

            N = int(yv_dec_all.numel())
            msg = (
                f"[VAL EPOCH {epoch_num}] "
                f"MAE={overall_mae:.6f} "
                f"RMSE={overall_rmse:.6f} "
                f"MSE={overall_mse:.6f} "
                f"QLIKE={overall_qlike:.6f} "
                + (f"| ACC={acc:.3f} " if acc is not None else "")
                + f"| N={N}"
            )
            # Append combined loss to msg
            try:
                # reuse the same composition for display
                _dir_display = None
                if yd_cpu is not None and pd_cpu is not None and yd_cpu.numel() > 0 and pd_cpu.numel() > 0:
                    pt_disp = pd_cpu
                    if torch.isfinite(pt_disp).any() and (pt_disp.min() < 0 or pt_disp.max() > 1):
                        _dir_display = float(F.binary_cross_entropy_with_logits(pt_disp, (yd_cpu.float()*0.9+0.05)).item())
                    else:
                        _dir_display = float(F.binary_cross_entropy(torch.clamp(pt_disp,1e-7,1-1e-7), (yd_cpu.float()*0.9+0.05)).item())
                _val_loss_disp = (overall_mae + overall_rmse) + (0.05 * _dir_display if _dir_display is not None else 0.0)
                msg += f"| VAL_LOSS={_val_loss_disp:.6f}"
            except Exception:
                pass
            print(msg)

            # Expose to callback metrics for progress bar if desired
            try:
                trainer.callback_metrics["val_mae_overall"] = torch.tensor(float(overall_mae))
                trainer.callback_metrics["val_rmse_overall"] = torch.tensor(float(overall_rmse))
                trainer.callback_metrics["val_mse_overall"] = torch.tensor(float(overall_mse))
                trainer.callback_metrics["val_qlike_overall"] = torch.tensor(float(overall_qlike))
                if acc is not None:
                    trainer.callback_metrics["val_acc_overall"] = torch.tensor(float(acc))
                trainer.callback_metrics["val_N_overall"] = torch.tensor(float(N))
            except Exception:
                pass
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
                "overall": {k: v for k, v in overall.items() if k in ("mae","rmse","mse","qlike","val_loss","dir_bce")},
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
            
class LearnableMultiTaskLoss(nn.Module):
    """
    Two-loss combiner with learnable uncertainty-style weights.
    total = exp(-s1) * L1 + exp(-s2) * L2 + (s1 + s2)

    Accepts predictions in **either** form:
      • list/tuple: [pred_vol, pred_dir]
      • stacked tensor: [..., 2, K] where index 0 is vol-quantiles (K), index 1 is
        the direction logit replicated across the same K dimension.

    Targets can be [..., 2] (vol, dir) or provided as a list/tuple of tensors.
    """

    def __init__(self, loss_vol: nn.Module, loss_dir: nn.Module, init_weights=(1.0, 0.5)):
        super().__init__()
        self.loss_vol = loss_vol
        self.loss_dir = loss_dir
        w1, w2 = float(init_weights[0]), float(init_weights[1])
        self.s = nn.Parameter(torch.tensor([
            -np.log(max(w1, 1e-8)),
            -np.log(max(w2, 1e-8)),
        ], dtype=torch.float32))

    @staticmethod
    def _ensure_tensor(x):
        """Convert any nested list/tuple of tensors into a single torch.Tensor."""
        if torch.is_tensor(x):
            return x
        if isinstance(x, (list, tuple)):
            flat = [LearnableMultiTaskLoss._ensure_tensor(t) for t in x]
            try:
                return torch.stack(flat, dim=-1)  # stack into last (quantile) dim
            except RuntimeError:
                return flat[0]
        raise TypeError(f"Expected Tensor/List/Tuple, got {type(x)}")

    def _unpack_targets(self, target):
        # Accept missing direction target; return (yv, yd_or_none)
        if torch.is_tensor(target):
            if target.ndim == 3 and target.size(-1) >= 1:
                yv = target[..., 0]
                yd = target[..., 1] if target.size(-1) > 1 else None
                return yv, yd
            if target.ndim == 2 and target.size(-1) >= 1:
                yv = target[:, 0]
                yd = target[:, 1] if target.size(-1) > 1 else None
                return yv, yd
            raise ValueError(f"Unexpected target shape: {target.shape}")
        if isinstance(target, (list, tuple)) and len(target) >= 1:
            yv = target[0]
            yd = target[1] if len(target) > 1 else None
            if torch.is_tensor(yv) and yv.ndim >= 3 and yv.size(-1) == 1:
                yv = yv[..., 0]
            if isinstance(yd, torch.Tensor) and yd.ndim >= 3 and yd.size(-1) == 1:
                yd = yd[..., 0]
            return yv, yd
        raise TypeError("LearnableMultiTaskLoss expects target with at least one channel or a list/tuple")

    def forward(self, y_pred, target, **kwargs):
        # ---- parse predictions (list/tuple or stacked tensor) ----
        if isinstance(y_pred, (list, tuple)):
            p_vol = self._ensure_tensor(y_pred[0])
            p_dir = self._ensure_tensor(y_pred[1]) if len(y_pred) > 1 else None
        elif torch.is_tensor(y_pred):
            t = y_pred
            # squeeze pred_len dim if present (…, 1, 2, K) -> (…, 2, K)
            if t.ndim >= 4 and t.size(-3) == 1:
                t = t.squeeze(-3)
            if t.ndim >= 3 and t.size(-2) >= 2:
                p_vol = t.select(dim=-2, index=0)  # [..., K]
                p_dir = t.select(dim=-2, index=1)  # [..., K] replicated
            else:
                p_vol = t
                p_dir = None
        else:
            raise TypeError("LearnableMultiTaskLoss expects y_pred as list/tuple or stacked tensor")

        # ---- targets ----
        yv, yd = self._unpack_targets(target)

        # ---- align quantile dim for vol head ----
        try:
            q = len(getattr(self.loss_vol, "quantiles", []))
        except Exception:
            q = None
        if q and torch.is_tensor(p_vol) and p_vol.ndim >= 2 and p_vol.size(-1) != q:
            if p_vol.size(-1) == 1:
                p_vol = p_vol.repeat_interleave(q, dim=-1)
            else:
                p_vol = p_vol.mean(dim=-1, keepdim=True).repeat_interleave(q, dim=-1)

        # ---- collapse direction to a single logit if it has a bogus K dim ----
        if p_dir is not None and torch.is_tensor(p_dir) and p_dir.ndim >= 2 and p_dir.size(-1) > 1:
            mid = p_dir.size(-1) // 2
            p_dir = p_dir.index_select(-1, torch.tensor([mid], device=p_dir.device)).squeeze(-1)

        # ---- compute component losses ----
        L1 = self.loss_vol(p_vol, yv)
        if hasattr(L1, "mean"):
            L1 = L1.mean()

        if isinstance(yd, torch.Tensor) and yd.numel() > 0 and p_dir is not None:
            L2 = self.loss_dir(p_dir, yd)
            if hasattr(L2, "mean"):
                L2 = L2.mean()
        else:
            L2 = torch.zeros((), device=p_vol.device, dtype=p_vol.dtype)

        # ---- cache internals for callbacks ----
        self.last_L1 = torch.as_tensor(L1).detach()
        self.last_L2 = torch.as_tensor(L2).detach() if isinstance(L2, torch.Tensor) else torch.tensor(float(L2))
        self.last_s  = self.s.detach()
        self.last_w  = torch.exp(-self.s).detach()

        weights = torch.exp(-self.s)
        return weights[0] * L1 + weights[1] * L2 + self.s.sum()

    # ---- PF hooks ----
    def rescale_parameters(self, y_pred, **kwargs):
        # support both forms during any PF-internal rescaling
        if isinstance(y_pred, (list, tuple)) and len(y_pred) >= 1:
            vol = self._ensure_tensor(y_pred[0])
            dir_ = self._ensure_tensor(y_pred[1]) if len(y_pred) > 1 else None
            if hasattr(self.loss_vol, "rescale_parameters"):
                vol = self.loss_vol.rescale_parameters(vol, **kwargs)
            return [vol, dir_]
        return y_pred

    def to_prediction(self, y_pred, normalize: bool = False, **kwargs):
        """Return point predictions for both heads, robust to PF version drift.
        Handles list/tuple or stacked-tensor heads and aligns the quantile
        dimension before delegating to the underlying loss (if available).
        """
        import torch as _torch

        def _split_heads(t):
            # Accept [vol, dir] or stacked tensor [..., n_targets, K]
            if isinstance(t, (list, tuple)):
                vol = self._ensure_tensor(t[0])
                dir_ = self._ensure_tensor(t[1]) if len(t) > 1 else None
                return vol, dir_
            if _torch.is_tensor(t):
                # (…, 1, 2, K) -> (…, 2, K)
                if t.ndim >= 4 and t.size(-3) == 1:
                    t = t.squeeze(-3)
                if t.ndim >= 3 and t.size(-2) >= 2:
                    vol = t.select(dim=-2, index=0)
                    dir_ = t.select(dim=-2, index=1)
                    return vol, dir_
                return t, None
            return t, None

        def _median_quantile(t):
            t = self._ensure_tensor(t)
            if t.ndim == 3 and t.size(1) == 1:
                t = t.squeeze(1)
            if t.ndim >= 2 and t.size(-1) > 1:
                return t[..., t.size(-1) // 2]
            return t.squeeze(-1)

        vol, dir_ = _split_heads(y_pred)

        # Align quantiles with configured loss before delegating
        try:
            q = len(getattr(self.loss_vol, "quantiles", []))
        except Exception:
            q = None
        if q and _torch.is_tensor(vol) and vol.ndim >= 2 and vol.size(-1) != q:
            if vol.size(-1) == 1:
                vol = vol.repeat_interleave(q, dim=-1)
            else:
                vol = vol.mean(dim=-1, keepdim=True).repeat_interleave(q, dim=-1)

        if hasattr(self.loss_vol, "to_prediction"):
            try:
                vol_out = self.loss_vol.to_prediction(vol, normalize=normalize, **kwargs)
                return [vol_out, dir_]
            except TypeError:
                try:
                    vol_out = self.loss_vol.to_prediction(vol, **kwargs)
                    return [vol_out, dir_]
                except Exception:
                    pass
            except Exception:
                pass

        return [_median_quantile(vol), dir_]

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
    """Return stacked tensor [..., 2, K] where:
       index 0 = volatility quantiles (K=7), index 1 = direction logits (replicated across K)."""
    def __init__(self, vol_head: nn.Module, dir_head: nn.Module):
        super().__init__()
        self.vol = vol_head
        self.dir = dir_head
    def forward(self, x):
        vol_out = self.vol(x)                 # [..., 7]
        dir_out = self.dir(x)                 # [..., 1] (logit)
        dir_rep = dir_out.repeat_interleave(vol_out.size(-1), dim=-1)  # [..., 7]
        return torch.stack([vol_out, dir_rep], dim=-2)  # [..., 2, 7]








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
# Permutation Importance helpers at module scope (decoded metric = MAE + RMSE + 0.05 * DirBCE)
# -----------------------------------------------------------------------

def _extract_norm_from_dataset(ds: TimeSeriesDataSet):
    """Return the GroupNormalizer used for realised_vol in our MultiNormalizer."""
    try:
        norm = ds.get_parameters()["target_normalizer"]
        if hasattr(norm, "normalizers") and len(norm.normalizers) >= 1:
            return norm.normalizers[0]
    except Exception:
        pass
    return None



def _evaluate_decoded_metrics(
    model,
    ds: TimeSeriesDataSet,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    prefetch: int,
    pin_memory: bool,
):
    """
    Compute decoded MAE, RMSE, DirBCE and combined val_loss on up to max_batches.
    Returns: (mae, rmse, dir_bce, val_loss, n)
    """
    vol_norm = _extract_norm_from_dataset(ds)

    dl = ds.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
    )

    # Get model device (first parameter device)
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device("cpu")

    model.eval()
    y_all, p_all, yd_all, pd_all, g_all = [], [], [], [], []
    skipped_reasons = {"no_groups": 0, "no_targets": 0, "bad_pred_format": 0}
    with torch.no_grad():
        for b_idx, batch in enumerate(dl):
            if max_batches is not None and b_idx >= int(max_batches):
                break
            # Move all tensors in batch to the model's device
            batch = tuple(t.to(model_device) if torch.is_tensor(t) else t for t in batch)
            
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                x, y = batch, None

            if not isinstance(x, dict):
                skipped_reasons["bad_pred_format"] += 1
                continue

            # ---- groups (robust; avoid `or` with tensors) ----
            g = x.get("groups", None)
            if g is None:
                g = x.get("group_ids", None)
            if isinstance(g, (list, tuple)):
                g = g[0] if len(g) > 0 else None
            if torch.is_tensor(g) and g.ndim > 1 and g.size(-1) == 1:
                g = g.squeeze(-1)
            if not torch.is_tensor(g):
                skipped_reasons["no_groups"] += 1
                continue

            # ---- targets (robust; avoid `or` with tensors) ----
            # ---- targets (robust; handles tensor OR list/tuple; avoids truthiness on tensors) ----
            y_vol, y_dir = None, None

            def _extract_targets(obj):
                """Return (y_vol, y_dir) from PF-style targets."""
                if torch.is_tensor(obj):
                    t = obj
                    # PF often gives [B, 1, n_targets] for decoder_target
                    if t.ndim == 3 and t.size(-1) == 1:
                        t = t[..., 0]  # -> [B, n_targets]
                    # Also seen: [B, 1, n_targets] (pred_len=1) but with the middle dim as length 1
                    if t.ndim == 3 and t.size(1) == 1:
                        t = t[:, 0, :]  # -> [B, n_targets]
                    if t.ndim == 2 and t.size(1) >= 1:
                        yv = t[:, 0]
                        yd = t[:, 1] if t.size(1) > 1 else None
                        return yv, yd
                    return None, None

                if isinstance(obj, (list, tuple)) and len(obj) >= 1:
                    yv = obj[0]
                    yd = obj[1] if len(obj) > 1 else None
                    # yv may itself be [B, 1, 1] or [B, 1]
                    if torch.is_tensor(yv):
                        if yv.ndim == 3 and yv.size(-1) == 1:
                            yv = yv[..., 0]
                        if yv.ndim == 3 and yv.size(1) == 1:
                            yv = yv[:, 0, :]
                        if yv.ndim == 2 and yv.size(-1) == 1:
                            yv = yv[:, 0]
                    if torch.is_tensor(yd):
                        if yd.ndim == 3 and yd.size(-1) == 1:
                            yd = yd[..., 0]
                        if yd.ndim == 3 and yd.size(1) == 1:
                            yd = yd[:, 0, :]
                        if yd.ndim == 2 and yd.size(-1) == 1:
                            yd = yd[:, 0]
                    return yv if torch.is_tensor(yv) else None, yd if torch.is_tensor(yd) else None

                return None, None

            # Preferred: decoder_target; else target
            dec_t = x.get("decoder_target", None)
            if dec_t is None:
                dec_t = x.get("target", None)

            if dec_t is not None:
                y_vol, y_dir = _extract_targets(dec_t)

            # Fallback: dataloader second item (PF often provides batch[1] = targets)
            if (y_vol is None or (y_dir is None and y is not None)) and y is not None:
                yv2, yd2 = _extract_targets(y)
                if y_vol is None:
                    y_vol = yv2
                if y_dir is None and yd2 is not None:
                    y_dir = yd2

            if not torch.is_tensor(y_vol):
                skipped_reasons["no_targets"] += 1
                continue

            # ---- forward pass (move to model device) ----
            x_dev = {}
            for k, v in x.items():
                if torch.is_tensor(v):
                    x_dev[k] = v.to(model_device, non_blocking=True)
                elif isinstance(v, (list, tuple)):
                    x_dev[k] = [vv.to(model_device, non_blocking=True) if torch.is_tensor(vv) else vv for vv in v]
                else:
                    x_dev[k] = v

            y_hat = model(x_dev)
            pred = getattr(y_hat, "prediction", y_hat)
            if isinstance(pred, dict) and "prediction" in pred:
                pred = pred["prediction"]

            # ---- extract heads ----
            # ---- extract heads (robust; mirrors callback) ----
            def _to_median_q(t):
                if t is None:
                    return None
                if torch.is_tensor(t):
                    # squeeze pred_len dim if present
                    if t.ndim >= 4 and t.size(1) == 1:
                        t = t.squeeze(1)
                    if t.ndim == 3 and t.size(1) == 1:
                        t = t.squeeze(1)  # [B, K] or [B, 1]
                    if t.ndim == 2 and t.size(-1) >= 1:
                        return t[:, t.size(-1) // 2]  # median quantile
                    if t.ndim == 1:
                        return t
                return t

            def _to_dir_logit(t):
                if t is None:
                    return None
                if torch.is_tensor(t):
                    # squeeze pred_len dim if present
                    if t.ndim >= 4 and t.size(1) == 1:
                        t = t.squeeze(1)
                    if t.ndim == 3 and t.size(1) == 1:
                        t = t.squeeze(1)  # [B, K] or [B, 1]
                    if t.ndim == 2:
                        # if replicated across K, take middle slot; else squeeze singleton
                        if t.size(-1) > 1:
                            return t[:, t.size(-1) // 2]
                        return t.squeeze(-1)
                    if t.ndim == 1:
                        return t
                return t

            p_vol, p_dir = None, None
            if isinstance(pred, (list, tuple)):
                p_vol = _to_median_q(pred[0])
                p_dir = _to_dir_logit(pred[1] if len(pred) > 1 else None)
            elif torch.is_tensor(pred):
                t = pred
                # squeeze pred_len dim if present: [B, 1, 2, K] -> [B, 2, K]
                if t.ndim >= 4 and t.size(1) == 1:
                    t = t.squeeze(1)
                if t.ndim == 3 and t.size(1) >= 2:
                    vol = t[:, 0, :]  # [B, K]
                    d   = t[:, 1, :]  # [B, K] (replicated logits) or [B, 1]
                    p_vol = _to_median_q(vol)
                    p_dir = _to_dir_logit(d)
                elif t.ndim >= 2:
                    p_vol = _to_median_q(t)
                    p_dir = None
                else:
                    skipped_reasons["bad_pred_format"] += 1
                    continue
            else:
                skipped_reasons["bad_pred_format"] += 1
                continue

            if not torch.is_tensor(p_vol) or not torch.is_tensor(g):
                skipped_reasons["bad_pred_format"] += 1
                continue

            # collapse dir head to single logit if needed
            if p_dir is not None and torch.is_tensor(p_dir) and p_dir.ndim >= 2 and p_dir.size(-1) > 1:
                mid = p_dir.size(-1) // 2
                p_dir = p_dir.index_select(-1, torch.tensor([mid], device=p_dir.device)).squeeze(-1)

            # decode back to physical scale (vol only)
            try:
                y_dec = vol_norm.decode(y_vol.to(model_device).unsqueeze(-1), group_ids=g.to(model_device).unsqueeze(-1)).squeeze(-1)
                p_dec = vol_norm.decode(p_vol.to(model_device).unsqueeze(-1), group_ids=g.to(model_device).unsqueeze(-1)).squeeze(-1)
            except Exception:
                y_dec, p_dec = y_vol.to(model_device), p_vol.to(model_device)

            y_all.append(y_dec.detach().cpu())
            p_all.append(p_dec.detach().cpu())
            g_all.append(g.detach().cpu())

            # collect direction only if both available
            if torch.is_tensor(y_dir) and torch.is_tensor(p_dir):
                yd_flat = y_dir.reshape(-1)
                pd_flat = p_dir.reshape(-1)
                L2 = min(yd_flat.numel(), pd_flat.numel())
                if L2 > 0:
                    yd_all.append(yd_flat[:L2].detach().cpu())
                    pd_all.append(pd_flat[:L2].detach().cpu())

    if not y_all:
        print("[FI] No valid batches produced targets/predictions; dataset may be empty or shapes unexpected.")
        print(f"[FI] Skips — groups:{skipped_reasons['no_groups']}, targets:{skipped_reasons['no_targets']}, pred:{skipped_reasons['bad_pred_format']}")
        return float("nan"), float("nan"), float("nan"), float("nan"), 0

    y = torch.cat(y_all)
    p = torch.cat(p_all)
    mae = (p - y).abs().mean().item()
    rmse = torch.sqrt(((p - y) ** 2).mean()).item()

    # Direction BCE (logits-aware) with label smoothing 0.1
    dir_bce = float("nan")
    if yd_all and pd_all:
        yd = torch.cat(yd_all).float()
        pd = torch.cat(pd_all)
        try:
            if torch.isfinite(pd).any() and (pd.min() < 0 or pd.max() > 1):
                yt_s = yd * 0.9 + 0.05
                dir_bce_t = F.binary_cross_entropy_with_logits(pd, yt_s)
            else:
                pr = torch.clamp(pd, 1e-7, 1 - 1e-7)
                yt_s = yd * 0.9 + 0.05
                dir_bce_t = F.binary_cross_entropy(pr, yt_s)
            dir_bce = float(dir_bce_t.item())
        except Exception:
            pass

    val_loss = (float(mae) + float(rmse)) + (0.05 * dir_bce if np.isfinite(dir_bce) else 0.0)
    return float(mae), float(rmse), float(dir_bce), float(val_loss), int(y.numel())



def run_permutation_importance(
    model,
    base_df: pd.DataFrame,
    build_ds_fn,
    features: List[str],
    block_size: int,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    prefetch: int,
    pin_memory: bool,
    out_csv_path: str,
    uploader,
) -> None:
    """
    Compute FI by permuting each feature and measuring Δ(val_loss) where
    val_loss = MAE + RMSE + 0.05 * DirBCE (decoded scale).
    Saves CSV with: feature, baseline_val_loss, permuted_val_loss, delta, mae, rmse, dir_bce, n.
    """
    ds_base = build_ds_fn(base_df, predict=False)
    max_batches = 24
    try:
        print(f"[FI] Dataset size (samples): {len(ds_base)} | batch_size={batch_size}")
    except Exception:
        pass
    b_mae, b_rmse, b_dir, b_val, n = _evaluate_decoded_metrics(
        model, ds_base, batch_size, max_batches, num_workers, prefetch, pin_memory
    )
    print(f"[FI] Baseline val_loss = {b_val:.6f} (MAE={b_mae:.6f}, RMSE={b_rmse:.6f}, DirBCE={b_dir:.6f}) | N={n}")

    rows = []
    for feat in features:
        if feat not in base_df.columns:
            print(f"[FI] Skipping missing feature: {feat}")
            continue
        df_p = base_df.copy()
        _permute_series_inplace(df_p, feat, block=block_size, group_col=GROUP_ID[0] if GROUP_ID else "asset")
        ds_p = build_ds_fn(df_p, predict=False)
        p_mae, p_rmse, p_dir, p_val, n_p = _evaluate_decoded_metrics(
            model, ds_p, batch_size, max_batches, num_workers, prefetch, pin_memory
        )
        delta = (p_val - b_val) if (np.isfinite(p_val) and np.isfinite(b_val)) else float("nan")
        print(f"[FI] {feat:>20} | val_p={p_val:.6f} | Δ={delta:.6f} | (MAE={p_mae:.6f}, RMSE={p_rmse:.6f}, DirBCE={p_dir:.6f})")
        rows.append({
            "feature": feat,
            "baseline_val_loss": b_val,
            "permuted_val_loss": p_val,
            "delta": delta,
            "mae": p_mae,
            "rmse": p_rmse,
            "dir_bce": p_dir,
            "n": n_p,
        })

    try:
        _df = pd.DataFrame(rows).sort_values("delta", ascending=False)
        _df.to_csv(out_csv_path, index=False)
        print(f"✓ Saved FI → {out_csv_path}")
        try:
            uploader(out_csv_path, f"{GCS_OUTPUT_PREFIX}/" + os.path.basename(out_csv_path))
        except Exception as e:
            print(f"[WARN] Could not upload FI CSV: {e}")
    except Exception as e:
        print(f"[WARN] Failed to save FI: {e}")
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
    
    es_cb = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOP_PATIENCE,
    mode="min"
    )

    bar_cb = TQDMProgressBar()

    metrics_cb = PerAssetMetrics(
        id_to_name=rev_asset,
        vol_normalizer=_extract_norm_from_dataset(training_dataset)
    )

    # If you have a custom checkpoint mirroring callback
    mirror_cb = MirrorCheckpoints()
    from pytorch_forecasting.metrics import MultiLoss

    # Fixed weights
    FIXED_VOL_WEIGHT = 1.0
    FIXED_DIR_WEIGHT = 0.1

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
                underestimation_factor=1.115,  # keep asymmetric penalty
            ),
            LabelSmoothedBCE(smoothing=0.1),
        ], weights=[FIXED_VOL_WEIGHT, FIXED_DIR_WEIGHT]),
        logging_metrics=[],
        log_interval=50,
        log_val_interval=10,
        reduce_on_plateau_patience=5,
        reduce_on_plateau_min_lr=1e-5,
    )

        # ----------------------------
    # Create callbacks BEFORE Trainer
    # ----------------------------
    lr_cb = LearningRateMonitor(logging_interval="step")

    best_ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename=f"tft_best_e{MAX_EPOCHS}_{RUN_SUFFIX}",
        dirpath=str(LOCAL_CKPT_DIR),
    )


    # ----------------------------
    # Trainer instance
    # ----------------------------
    trainer = Trainer(
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        num_sanity_val_steps = 0,
        logger=logger,
        callbacks=[lr_cb, best_ckpt_cb, es_cb, bar_cb, metrics_cb, mirror_cb],
        check_val_every_n_epoch=int(ARGS.check_val_every_n_epoch),
        log_every_n_steps=int(ARGS.log_every_n_steps),
    )


    from types import MethodType

    # Completely skip PF's internal figure creation & TensorBoard logging during validation.
    def _no_log_prediction(self, *args, **kwargs):
        # Intentionally do nothing so BaseModel.create_log won't attempt to log a Matplotlib figure.
        return

    tft.log_prediction = MethodType(_no_log_prediction, tft)

    # (Optional, extra safety) If any PF path calls plot_prediction directly, hand back a blank figure
    # so nothing touches your tensors or bf16 -> NumPy conversion.
    def _blank_plot(self, *args, **kwargs):
        import matplotlib
        matplotlib.use("Agg", force=True)  # headless-safe backend
        import matplotlib.pyplot as plt
        fig = plt.figure()
        return fig

    tft.plot_prediction = MethodType(_blank_plot, tft)

    # Train the model
    trainer.fit(tft, train_dataloader, val_dataloader)

    # Run FI permutation testing if enabled
    if ENABLE_FEATURE_IMPORTANCE:
        fi_csv = str(LOCAL_OUTPUT_DIR / f"tft_perm_importance_e{MAX_EPOCHS}_{RUN_SUFFIX}.csv")
        feats = time_varying_unknown_reals.copy()
        # (optional) drop calendar features from FI to focus on learned signals
        feats = [f for f in feats if f not in ("sin_tod", "cos_tod", "sin_dow", "cos_dow", "Is_Weekend")]
        run_permutation_importance(
            model=tft,
            base_df=val_df,
            build_ds_fn=build_dataset,
            features=feats,
            block_size=int(PERM_BLOCK_SIZE) if PERM_BLOCK_SIZE else 1,
            batch_size=batch_size,
            max_batches=int(FI_MAX_BATCHES) if FI_MAX_BATCHES else 20,
            num_workers=worker_cnt,
            prefetch=prefetch,
            pin_memory=pin,
            out_csv_path=fi_csv,
            uploader=upload_file_to_gcs,
        )
    # --- Safe plotting/logging: deep-cast any nested tensors to CPU float32 ---
    # --- Safe plotting/logging: class-level patch to handle bf16 + integer lengths robustly ---

    from pytorch_forecasting.models.base._base_model import BaseModel # type: ignore

    def _deep_cpu_float(x):
        if torch.is_tensor(x):
            # keep integer tensors as int64; cast others to float32 for matplotlib
            if x.dtype in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
                getattr(torch, "long", torch.int64)
            ):
                return x.detach().to(device="cpu", dtype=torch.int64)
            return x.detach().to(device="cpu", dtype=torch.float32)
        if isinstance(x, list):
            return [_deep_cpu_float(v) for v in x]
        if isinstance(x, tuple):
            casted = tuple(_deep_cpu_float(v) for v in x)
            try:
                # preserve namedtuple types
                return x.__class__(*casted)
            except Exception:
                return casted
        if isinstance(x, dict):
            return {k: _deep_cpu_float(v) for k, v in x.items()}
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
            if np.issubdtype(x.dtype, np.integer):
                return x.astype(np.int64, copy=False)
            return x.astype(np.float32, copy=False)
        return x

    def _to_numpy_int64_array(v):
        if torch.is_tensor(v):
            return v.detach().cpu().long().numpy()
        if isinstance(v, np.ndarray):
            return v.astype(np.int64, copy=False)
        if isinstance(v, (list, tuple)):
            out = []
            for el in v:
                if torch.is_tensor(el):
                    out.append(int(el.detach().cpu().item()))
                else:
                    out.append(int(el))
            return np.asarray(out, dtype=np.int64)
        if isinstance(v, (int, np.integer)):
            return np.asarray([int(v)], dtype=np.int64)
        return v  # leave unknowns as-is

    def _fix_lengths_in_x(x):
        # PF expects max() & python slicing with these; make them numpy int64 arrays
        if isinstance(x, dict):
            for key in ("encoder_lengths", "decoder_lengths"):
                if key in x and x[key] is not None:
                    x[key] = _to_numpy_int64_array(x[key])
        return x

