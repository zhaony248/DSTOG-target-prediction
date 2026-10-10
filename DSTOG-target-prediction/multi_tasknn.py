import torch
import numpy as np
import pandas as pd
import random
from lightning import pytorch as pl
from chemprop import data, models, nn
from sklearn.metrics import roc_auc_score

torch.set_float32_matmul_precision('high')

# ==========================================
# 1. Configuration
# ==========================================
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Curated training set (not included in this repository): a 'smiles' column
# plus six target label columns named after the tasks below. Unlabeled
# compound-target pairs are left as NaN.
INPUT_PATH = "training_data.csv"
CHECKPOINT_PATH = "multitask_classification_model.cpt"
BATCH_SIZE = 64
NUM_WORKERS = 4

# Six candidate targets of the TLR7 signalling pathway. The training set was
# curated from public bioactivity resources and supplemented with
# literature-derived data (see README).
TARGET_NAMES = ["TRAF6", "NFKB1", "NFKBIA", "MAP3K7", "JUN", "IRAK1"]
target_columns = TARGET_NAMES

# ==========================================
# 2. Data loading and processing
# ==========================================
df_input = pd.read_csv(INPUT_PATH)
smiles_column = 'smiles'

print(f"Dataset: {len(df_input)} compounds, {len(target_columns)} targets")

datapoints = []
for _, row in df_input.iterrows():
    smi = row[smiles_column]
    if pd.isna(smi):
        continue
    # Labels are passed as raw values; missing labels (NaN) are masked
    # automatically by chemprop during training.
    ys = row[target_columns].values.astype(np.float32)
    datapoints.append(data.MoleculeDatapoint.from_smi(smi, y=ys))

random.shuffle(datapoints)
n = len(datapoints)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
train_data = datapoints[:n_train]
val_data = datapoints[n_train:n_train + n_val]
test_data = datapoints[n_train + n_val:]

train_dset = data.MoleculeDataset(train_data)
val_dset = data.MoleculeDataset(val_data)
test_dset = data.MoleculeDataset(test_data)

train_loader = data.build_dataloader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = data.build_dataloader(val_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = data.build_dataloader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ==========================================
# 3. Model
# ==========================================
message_passing = nn.BondMessagePassing(d_h=500, depth=3, dropout=0.1)
agg = nn.MeanAggregation()
ffn = nn.BinaryClassificationFFN(
    n_tasks=len(target_columns),
    input_dim=message_passing.output_dim,
    hidden_dim=300,
    n_layers=2,
    dropout=0.1
)

model = models.MPNN(message_passing, agg, ffn, metrics=[])

# ==========================================
# 4. Training
# ==========================================
trainer = pl.Trainer(
    max_epochs=200,
    accelerator='auto',
    devices='auto',
    precision=32,
    num_sanity_val_steps=0,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')
    ]
)

print("Training...")
trainer.fit(model, train_loader, val_loader)
trainer.save_checkpoint(CHECKPOINT_PATH)


# ==========================================
# 5. Evaluation on the held-out test set
# ==========================================
def evaluate_manual():
    print("\n" + "=" * 40)
    print("Test-set metrics:")
    model.eval()

    test_results = trainer.predict(model, test_loader)
    all_preds = torch.cat(test_results, dim=0)
    pred_probs = torch.sigmoid(all_preds).cpu().numpy()

    true_values = np.array([dp.y for dp in test_data])

    for i, target in enumerate(target_columns):
        y_true_raw = true_values[:, i]
        mask_idx = ~np.isnan(y_true_raw)  # skip unlabeled compound-target pairs

        y_true = y_true_raw[mask_idx]
        y_prob = pred_probs[mask_idx, i]

        if len(y_true) > 0 and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            print(f"Target {target:8s} | AUC: {auc:.4f} | valid samples: {len(y_true)}")
        else:
            reason = "single-class samples only" if len(y_true) > 0 else "no valid test samples"
            print(f"Target {target:8s} | {reason}, AUC skipped")


if __name__ == "__main__":
    evaluate_manual()
