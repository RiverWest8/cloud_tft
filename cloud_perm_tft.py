import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lightning.pytorch import Trainer, seed_everything
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# Detect if running in Google Cloud environment
IN_CLOUD = os.getenv("CLOUD_ENV", "false").lower() == "true"

# Configure paths from environment variables or use defaults
HOME = Path.home()
TRAIN_P = Path(os.getenv("TRAIN_PATH", str(HOME / "Desktop" / "Data" / "CleanedData" / "universal_train.parquet")))
VAL_P   = Path(os.getenv("VAL_PATH", str(HOME / "Desktop" / "Data" / "CleanedData" / "universal_val.parquet")))
TEST_P  = Path(os.getenv("TEST_PATH", str(HOME / "Desktop" / "Data" / "CleanedData" / "universal_test.parquet")))

# -----------------------
# 0) Register asinh
# -----------------------
if ("asinh" not in GroupNormalizer.TRANSFORMATIONS
    or not isinstance(GroupNormalizer.TRANSFORMATIONS["asinh"], dict)):
    GroupNormalizer.TRANSFORMATIONS["asinh"] = {
        "forward": lambda x: torch.asinh(x) if torch.is_tensor(x) else np.arcsinh(x),
        "inverse": lambda x: torch.sinh(x)  if torch.is_tensor(x) else np.sinh(x),
    }
if hasattr(GroupNormalizer, "INVERSE_TRANSFORMATIONS"):
    GroupNormalizer.INVERSE_TRANSFORMATIONS.setdefault(
        "asinh", lambda x: torch.sinh(x) if torch.is_tensor(x) else np.sinh(x)
    )

# -----------------------
# 1) Paths & config
# -----------------------
GROUP_ID = ["asset"]
ENC_LEN = 24
PRED_LEN = 1
TRAIN_N = 5000
VAL_N = 1000
TEST_N = 1000
BATCH_SIZE = 256
MAX_EPOCHS = 10
LR = 2e-3  # start modest; raise if you want to overfit the 5k subset quickly

seed_everything(42)

# -----------------------
# 2) Load & basic prep
# -----------------------
def rs_variance(o, h, l, c):
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # standardize column names for target/class
    if "realised_vol" not in df.columns and "Realised_Vol" in df.columns:
        df = df.rename(columns={"Realised_Vol": "realised_vol"})
    if "direction" not in df.columns and "Direction" in df.columns:
        df = df.rename(columns={"Direction": "direction"})

    # compute realised_vol if missing
    if "realised_vol" not in df.columns:
        needed = {"Open","High","Low","Close"}
        if needed.issubset(df.columns):
            rv = rs_variance(df["Open"], df["High"], df["Low"], df["Close"])
            df["realised_vol"] = np.sqrt(np.clip(rv, 0.0, None)).shift(-1)
        else:
            raise ValueError("No realised_vol and missing OHLC columns to compute it.")

    # compute direction if missing (next-step up/down)
    if "direction" not in df.columns and {"Close"}.issubset(df.columns):
        df["direction"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # sort and create time_idx
    time_col = None
    for c in ["timestamp", "date", "Date", "time"]:
        if c in df.columns:
            time_col = c
            break
    if time_col:
        df = df.sort_values(GROUP_ID + [time_col], ascending=True)
    else:
        df = df.sort_values(GROUP_ID, ascending=True)
    df["time_idx"] = df.groupby(GROUP_ID).cumcount()

    # drop last row(s) with NaN targets due to shifting
    df = df.dropna(subset=["realised_vol"]).reset_index(drop=True)
    return df

train_df = load_and_prepare(TRAIN_P)
val_df   = load_and_prepare(VAL_P) if VAL_P.exists() else train_df.copy()
test_df  = load_and_prepare(TEST_P) if TEST_P.exists() else train_df.copy()

# take subsets
train_df = train_df.iloc[:TRAIN_N].copy()
val_df   = val_df.iloc[:VAL_N].copy()
test_df  = test_df.iloc[:TEST_N].copy()

print(f"[INFO] train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
print(f"[INFO] mean realised_vol(train)={train_df['realised_vol'].mean():.6g}")

# -----------------------
# 3) Build PF datasets
# -----------------------
# calendar/known features (minimal)
calendar_cols = []
time_varying_known_reals = calendar_cols + ["time_idx"]  # + any of your calendar cols
# numeric candidates
all_numeric = train_df.select_dtypes(include=[np.number]).columns.tolist()

# ensure realised_vol is NOT used as a feature (target only)
time_varying_unknown_reals = [
    c for c in all_numeric
    if c not in (calendar_cols + ["time_idx", "realised_vol"])
]

# (optional) remove realised_vol from the feature frame if you really want zero chance of leak
# NOTE: PF will still get the target from the 'target' argument, so we keep the column.
# If you insist on removing from the dataframe copy used for features, PF handles internally.

training_dataset = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="realised_vol",
    group_ids=GROUP_ID,
    min_encoder_length=ENC_LEN,
    max_encoder_length=ENC_LEN,
    min_prediction_length=PRED_LEN,
    max_prediction_length=PRED_LEN,
    static_categoricals=GROUP_ID,
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=time_varying_unknown_reals,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    target_normalizer=GroupNormalizer(
        groups=GROUP_ID,
        center=False,
        scale_by_group=True,
        transformation="asinh",
    ),
)

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, val_df, predict=False, stop_randomization=True
)
test_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset, test_df, predict=False, stop_randomization=True
)

train_loader = training_dataset.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
val_loader   = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

# -----------------------
# 4) Helpers: extract normalizer & safe decode
# -----------------------
def _extract_norm_from_dataset(ds: TimeSeriesDataSet):
    try:
        norm = ds.get_parameters()["target_normalizer"]
        # If it's a MultiNormalizer, take the first entry:
        if hasattr(norm, "normalizers") and len(norm.normalizers) >= 1:
            return norm.normalizers[0]
        return norm
    except Exception:
        return None

def safe_decode_vol(values: torch.Tensor, vol_norm, group_ids: torch.Tensor) -> torch.Tensor:
    """
    Try the normal decode path; fallback to identity if not available.
    values: [B,1] or [B]
    group_ids: [B,1] or [B]
    """
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    if group_ids.ndim == 1:
        group_ids = group_ids.unsqueeze(-1)
    try:
        out = vol_norm.decode(values, group_ids=group_ids)
        return out.squeeze(-1)
    except Exception:
        # last resort: identity (shouldn't happen if normalizer is from training_dataset)
        return values.squeeze(-1)

vol_norm = _extract_norm_from_dataset(training_dataset)

# -----------------------
# 5) Build TFT model
# -----------------------
tft = TemporalFusionTransformer.from_dataset(
    training_dataset,
    learning_rate=LR,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
    output_size=7,             # 7 quantiles
    loss=QuantileLoss(),       # head produces quantiles; we override training loss below
    log_interval=10,
    reduce_on_plateau_patience=4,
)
# attach normalizer so our custom steps can decode
tft.vol_norm = vol_norm

# -----------------------
# 6) Train on DECODED scale (override training/validation steps)
# -----------------------
from types import MethodType

def _extract_vol_pred_from_output(pred):
    """
    Accepts PF outputs in different shapes and returns a point vol forecast [B].
    We use mean-of-quantiles for stability.
    """
    if isinstance(pred, (list, tuple)):
        vol_q = pred[0]
        if torch.is_tensor(vol_q):
            if vol_q.ndim == 3 and vol_q.size(1) == 1:
                vol_q = vol_q[:, 0, :]  # [B, K]
            if vol_q.ndim == 2:
                return vol_q.mean(dim=-1)
    elif torch.is_tensor(pred):
        t = pred
        if t.ndim >= 4 and t.size(1) == 1:
            t = t.squeeze(1)
        if t.ndim == 3 and t.size(1) == 1:
            t = t[:, 0, :]
        if t.ndim == 2:
            D = t.size(-1)
            if D >= 2:
                vol_q = t[:, : D-1]
                return vol_q.mean(dim=-1)
    raise RuntimeError("Unexpected prediction structure for vol head")

def _coerce_first_target_to_1d(obj):
    """Return 1D tensor [B] for the first target (realised_vol)."""
    t = None
    if torch.is_tensor(obj):
        t = obj
    elif isinstance(obj, (list, tuple)):
        for it in obj:
            if torch.is_tensor(it):
                t = it
                break
    if t is None:
        return None
    # squeeze [B,1,K] -> [B,K], [B,K]->[B] first col
    if t.ndim == 3 and t.size(1) == 1:
        t = t[:, 0, :]
    if t.ndim == 3 and t.size(-1) == 1:
        t = t[..., 0]
    if t.ndim == 2 and t.size(1) >= 1:
        t = t[:, 0]
    if t.ndim != 1:
        return None
    return t

def _extract_y_vol_from_batch(x, y):
    # Prefer decoder_target/target stored in x dict
    for key in ("decoder_target", "target"):
        if isinstance(x, dict) and key in x:
            got = _coerce_first_target_to_1d(x[key])
            if got is not None:
                return got
    got = _coerce_first_target_to_1d(y)
    if got is not None:
        return got
    raise RuntimeError("Could not find realised_vol target in batch.")

def _extract_groups_from_batch(x):
    g_raw = None
    if isinstance(x, dict):
        g_raw = x.get("groups", None) or x.get("group_ids", None)
    g = None
    if torch.is_tensor(g_raw):
        g = g_raw
    elif isinstance(g_raw, (list, tuple)):
        for it in g_raw:
            if torch.is_tensor(it):
                g = it
                break
    if g is None:
        raise RuntimeError("Missing 'groups' in batch.")
    while g.ndim > 1 and g.size(-1) == 1:
        g = g.squeeze(-1)
    return g.long()

def _training_step_decoded(self, batch, batch_idx):
    x, y = batch
    out = self(x)
    pred = getattr(out, "prediction", out)
    p_vol = _extract_vol_pred_from_output(pred)
    y_vol = _extract_y_vol_from_batch(x, y)
    g     = _extract_groups_from_batch(x)

    # decode to physical units
    g_ids = g.to(self.device).unsqueeze(-1)
    y_dec = safe_decode_vol(y_vol.to(self.device).unsqueeze(-1), self.vol_norm, g_ids).squeeze(-1)
    p_dec = safe_decode_vol(p_vol.to(self.device).unsqueeze(-1), self.vol_norm, g_ids).squeeze(-1)
    p_dec = torch.clamp(p_dec, min=1e-8)

    mse = torch.mean((p_dec - y_dec) ** 2)
    mae = torch.mean(torch.abs(p_dec - y_dec))
    loss = mse + mae

    self.log("train_mae_dec", mae.detach(), prog_bar=True, on_step=False, on_epoch=True)
    self.log("train_rmse_dec", torch.sqrt(mse.detach()), prog_bar=False, on_step=False, on_epoch=True)
    self.log("train_loss_dec", loss.detach(), prog_bar=True, on_step=True, on_epoch=True)
    return loss

def _validation_step_decoded(self, batch, batch_idx):
    x, y = batch
    out = self(x)
    pred = getattr(out, "prediction", out)
    p_vol = _extract_vol_pred_from_output(pred)
    y_vol = _extract_y_vol_from_batch(x, y)
    g     = _extract_groups_from_batch(x)

    g_ids = g.to(self.device).unsqueeze(-1)
    y_dec = safe_decode_vol(y_vol.to(self.device).unsqueeze(-1), self.vol_norm, g_ids).squeeze(-1)
    p_dec = safe_decode_vol(p_vol.to(self.device).unsqueeze(-1), self.vol_norm, g_ids).squeeze(-1)
    p_dec = torch.clamp(p_dec, min=1e-8)

    mse = torch.mean((p_dec - y_dec) ** 2)
    mae = torch.mean(torch.abs(p_dec - y_dec))

    self.log("val_mae_dec", mae.detach(), prog_bar=True, on_step=False, on_epoch=True)
    self.log("val_rmse_dec", torch.sqrt(mse.detach()), prog_bar=True, on_step=False, on_epoch=True)
    return {"val_loss": (mse + mae).detach()}

tft.training_step  = MethodType(_training_step_decoded, tft)
tft.validation_step = MethodType(_validation_step_decoded, tft)

# -----------------------
# 7) Train
# -----------------------

def main():
    global MAX_EPOCHS, BATCH_SIZE, ENC_LEN
    parser = argparse.ArgumentParser(description="Train TFT on cloud perm data.")
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max_encoder_length", type=int, default=ENC_LEN)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--gcs_data_prefix", type=str, default="")
    parser.add_argument("--gcs_output_prefix", type=str, default="")
    args = parser.parse_args()

    # Override constants with CLI args
    MAX_EPOCHS = args.max_epochs
    BATCH_SIZE = args.batch_size
    ENC_LEN = args.max_encoder_length

    # Dataloaders with user-specified workers and prefetch
    global train_loader, val_loader
    train_loader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )
    val_loader = validation_dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        default_root_dir=args.gcs_output_prefix,
        enable_checkpointing=True
    )
    print("[INFO] CLI args:", vars(args))
    print("▶ Training TFT on decoded scale (MAE+MSE)…")
    trainer.fit(tft, train_loader, val_loader)


if __name__ == "__main__":
    main()

# Example CLI usage:
#   python cloud_perm_tft.py