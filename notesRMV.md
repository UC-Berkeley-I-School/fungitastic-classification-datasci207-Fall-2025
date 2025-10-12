Awesome—here’s a focused, practical game plan to move from **baseline → strong image model → image+metadata fusion → robust evaluation** on FungiTastic-Mini (closed-set).

---

# 1) Start with a solid image-only baseline

**Goal:** a clean reference to measure metadata lift against.

* **Backbone:** `efficientnet_b3` or `vit_base_patch16_224` (via `timm`). Start at **300p**; try **500p** if VRAM allows.
* **Transforms (train / val):**

  * Train: `Resize((S,S)) → RandomResizedCrop(S, scale=(0.8,1.0)) → RandomHorizontalFlip → ColorJitter (light) → ToTensor → Normalize(ImageNet)`
  * Val: `Resize((S,S)) → ToTensor → Normalize`
  * If your brightness EDA flagged issues: prefer **RandomBrightness/Contrast** jitter over heavy color jitter.
* **Optimization:** `AdamW(lr=1e-3, weight_decay=1e-4)`, **OneCycle** or **Cosine** LR; **label smoothing** (e.g., 0.05) helps.
* **Imbalance handling:** class weights in `CrossEntropyLoss` or `WeightedRandomSampler`. Compute from train counts.
* **Training details:** 10–20 epochs to start, mixed precision if available, early stopping on **val macro-F1**.
* **Metrics:** **Top-1 accuracy** + **macro-F1** (macro matters for the long tail). Also report **Top-5**.

> Minimal ablation: EfficientNet-B0/B3 at 300p vs 500p; augmentation on/off; class weights on/off.

---

# 2) Add metadata via late fusion (simple & strong)

**Idea:** keep the CNN, add a small MLP head that **concatenates image embedding + metadata vector**.

* **Prep metadata:** you already built `col_tf` → `Xtr_meta`, `Xva_meta`. Keep **fit on train only**.
* **Model:**

  1. CNN backbone with `num_classes=0` to output **image embedding** (`feat_dim`).
  2. Concatenate with **`meta`** (shape `[B, meta_dim]`), feed into a small **MLP head** → logits.
* **Training:** joint training end-to-end (backbone + head) with same loss/optimizer.
* **Checks:** ensure batch indices align with rows in `X_meta` (reset indices; pass batch indices from your dataset).

> Ablations: image-only vs image+**(month)** vs +**(month+habitat)** vs +**(month+habitat+lat/lon+elev)**. Report Δmacro-F1 vs baseline.

---

# 3) Simpler alternatives you can run fast

Useful when you’re iterating or low on compute.

* **Frozen embeddings + linear head:** freeze the CNN; train just a linear classifier (strong baseline).
* **Tabular-only probe:** train a tree model (RandomForest/LightGBM/CatBoost) on **metadata alone** → see standalone metadata signal.
* **Stacking:** train

  * (A) image-only logits → softmax
  * (B) metadata-only classifier
  * stack their outputs via a small logistic layer on val (meta-classifier of predictions).

---

# 4) Regularization & robustness (turn on as you scale)

* **Mixup/CutMix** (α≈0.2) can help long-tail; try after baseline is stable.
* **RandAugment/TrivialAugment** (light) can help generalization.
* **Exposures:** if many under/over-exposed images, add targeted jitter or discard extremes (from your brightness metrics) as an ablation.

---

# 5) Evaluation & reporting

* **Primary:** **macro-F1** and **Top-1** on **Val (Closed-set)**.
* **Diagnostics:** confusion matrix (top-k species), per-habitat error, per-month error, accuracy vs brightness bins.
* **Calibration (optional):** temperature scaling on val; report ECE if you’ll threshold/abstain later.

**Template table:**

| Model                    | Image size | Metadata     | Aug   | Class-bal. | Val Acc | Val macro-F1 | ΔF1 vs base |
| ------------------------ | ---------: | ------------ | ----- | ---------- | ------: | -----------: | ----------: |
| EfficientNet-B3          |        300 | –            | light | weights    |       … |            … |           – |
| + Month(sin,cos)         |        300 | time         | light | weights    |       … |            … |          +… |
| + Month+Habitat          |        300 | time+hab     | light | weights    |       … |            … |          +… |
| + Month+Hab+Lat/Lon/Elev |        300 | time+geo+hab | light | weights    |       … |            … |          +… |

---

# 6) Practical code skeletons (tiny)

**Class weights (PyTorch):**

```python
import numpy as np, torch
from collections import Counter

cls = train_df['species'].astype('category')
class_to_id = {c:i for i,c in enumerate(cls.cat.categories)}
y_train = train_df['species'].map(class_to_id).values
counts = Counter(y_train)
num_classes = len(counts)
freq = np.array([counts[i] for i in range(num_classes)])
weights = (freq.sum() / (num_classes * freq))
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
```

**Late fusion head:**

```python
import timm, torch.nn as nn, torch

backbone = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0).to(device)
feat_dim = backbone.num_features
meta_dim = Xtr_meta.shape[1]
num_classes = len(class_to_id)

head = nn.Sequential(
    nn.Linear(feat_dim + meta_dim, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, num_classes),
).to(device)

def forward(images, meta_np):
    feats = backbone(images)  # [B, feat_dim]
    meta = torch.from_numpy(meta_np).to(device).float()  # [B, meta_dim]
    return head(torch.cat([feats, meta], dim=1))
```

---

# 7) Run order (so you don’t get lost)

1. **Image-only baseline** @300p → metrics + saved checkpoint.
2. **Increase image strength**: minor aug / class weights → confirm gains.
3. **Late fusion v1**: add **month(sin,cos)** only.
4. **+Habitat** → **+Lat/Lon/Elev** → **+Substrate** (stop when gains plateau).
5. (Optional) Try 500p or a stronger backbone (e.g., ConvNeXt-Tiny, ViT-Base).
6. Add **calibration** and a short **error analysis** (by habitat/month/brightness).

---

# 8) Compute tips (macOS / general)

* macOS: use **`mps`** device (`torch.backends.mps.is_available()`), lower batch size if OOM.
* Save **best-val** weights; keep seeds fixed; log configs to a `runs/` folder.

---

If you want, I can turn this into a tiny **experiment harness** (one config dict → trains baseline or fusion and logs the results), or tailor the fusion code to your existing DataLoader so the metadata rows are indexed automatically per batch.


---
Awesome—let’s stand up a clean, **image-only baseline** you can run right now. I’ll give you the *why* and the *how* with copy-pasteable code.

---

# 1) Goal & setup

**Goal:** train a solid classifier using only the photos (no metadata) on **Mini-Train → Mini-ClosedSet-Val**.

**Backbone:** `efficientnet_b3` (great accuracy/speed tradeoff).
**Resolution:** start at **300 px** square (faster); bump to **500 px** later if you want more accuracy.

Install (once):

```bash
python -m pip install timm scikit-learn
```

---

# 2) Build a tiny Dataset from your DataFrame

We’ll read images using the `image_path` you already added and map `species → int` **using only train** (no leakage).

```python
import os, random, numpy as np, pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- Reproducibility
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- Choose label col
LABEL = "species" if "species" in train_df.columns else "category_id"

# --- Make label mapping from TRAIN ONLY
classes = pd.Index(sorted(train_df[LABEL].unique()))
cls2id  = {c:i for i,c in enumerate(classes)}
id2cls  = {i:c for c,i in cls2id.items()}

# --- Map labels to ids
train_df = train_df.copy()
val_df   = val_df.copy()
train_df["y"] = train_df[LABEL].map(cls2id)
val_df["y"]   = val_df[LABEL].map(cls2id)   # should be valid in closed-set; drop NaNs if any
val_df = val_df.dropna(subset=["y"]).copy()
train_df["y"] = train_df["y"].astype(int)
val_df["y"]   = val_df["y"].astype(int)

# --- PyTorch Dataset from a DataFrame
class DFImageDataset(Dataset):
    def __init__(self, df, img_col="image_path", y_col="y", transform=None):
        self.df = df.reset_index(drop=True)
        self.img_col, self.y_col = img_col, y_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[self.img_col]
        y = int(row[self.y_col])
        # load image
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y, path  # (tensor, int, path)
```

---

# 3) Transforms & DataLoaders

Use standard ImageNet normalization; light aug only on train.

```python
IMG_SIZE = 300  # try 500 later if VRAM allows

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = DFImageDataset(train_df, transform=train_tfms)
val_ds   = DFImageDataset(val_df,   transform=val_tfms)

# DataLoaders (tune batch_size to your device)
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
```

---

# 4) Model, loss, optimizer

* Device: use **MPS** on Apple GPU if available, else CUDA/CPU.
* Optional: **class weights** to help the long tail.

```python
import timm, torch
import torch.nn as nn

device = (
    torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Device:", device)

num_classes = len(classes)
model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes).to(device)

# --- (optional) class weights from train distribution
from collections import Counter
counts = Counter(train_df["y"].tolist())
freq = np.array([counts[i] for i in range(num_classes)], dtype=np.float32)
weights = (freq.sum() / (len(freq) * freq))  # inverse-frequency-ish
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)  # remove weight=... if you don't want it
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)  # 10 epochs start
```

---

# 5) Train/eval loops (with accuracy + macro-F1)

```python
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def run_epoch(loader, train=True):
    model.train(train)
    losses, ys, ps = [], [], []
    for imgs, y, _ in loader:
        imgs, y = imgs.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(imgs)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
        ps.append(logits.argmax(1).detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
    ytrue = np.concatenate(ys); ypred = np.concatenate(ps)
    acc = accuracy_score(ytrue, ypred)
    f1  = f1_score(ytrue, ypred, average="macro")
    return float(np.mean(losses)), acc, f1

best_f1 = -1.0
EPOCHS = 10  # start small; increase to ~20 once everything works

for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc, tr_f1 = run_epoch(train_loader, train=True)
    va_loss, va_acc, va_f1 = run_epoch(val_loader,   train=False)
    scheduler.step()

    print(f"Epoch {ep:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} f1 {tr_f1:.3f} "
          f"| val loss {va_loss:.3f} acc {va_acc:.3f} f1 {va_f1:.3f}")

    # Save best by val macro-F1
    if va_f1 > best_f1:
        best_f1 = va_f1
        os.makedirs("runs", exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "classes": classes.tolist(),
            "img_size": IMG_SIZE
        }, f"runs/effb3_imgonly_best.pt")
        print(f"  ↳ saved new best (val macro-F1={best_f1:.3f})")
```

---

# 6) Evaluate + basic diagnostics

Confusion matrix (top species) and a few worst errors help guide improvements.

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Collect predictions on val
ytrue_all, ypred_all = [], []
model.eval()
with torch.no_grad():
    for imgs, y, _ in val_loader:
        logits = model(imgs.to(device))
        ypred_all.append(logits.argmax(1).cpu().numpy())
        ytrue_all.append(y.numpy())
ytrue_all = np.concatenate(ytrue_all)
ypred_all = np.concatenate(ypred_all)

# Confusion on top-20 frequent classes in val
val_counts = pd.Series(ytrue_all).value_counts().head(20).index.tolist()
mask = np.isin(ytrue_all, val_counts)
cm = confusion_matrix(ytrue_all[mask], ypred_all[mask], labels=val_counts, normalize='true')

plt.figure(figsize=(8,6))
plt.imshow(cm, vmin=0, vmax=1)
plt.title("Confusion (top-20 classes, normalized)"); plt.colorbar()
ticks = [id2cls[i] for i in val_counts]
plt.xticks(range(len(ticks)), [t[:12] for t in ticks], rotation=90)
plt.yticks(range(len(ticks)), [t[:12] for t in ticks])
plt.tight_layout(); plt.show()
```

---

# 7) Why these choices (quick rationale)

* **EfficientNet-B3 @ 300px**: strong baseline with modest compute; upsize to 500px later for more detail.
* **Light aug**: keeps natural appearance but adds robustness (crop/flip).
* **Label smoothing + AdamW**: stabilizes training on fine-grained, long-tailed data.
* **Macro-F1**: balances head/tail species; use alongside accuracy.
* **Class weights**: optional but helpful if tail classes are tiny.

---

# 8) Next easy improvements

* Try **RandomBrightness/Contrast** if your brightness EDA showed exposure issues.
* Train **5–10 more epochs** with cosine LR for a bump.
* Swap backbone to `convnext_tiny` or raise `IMG_SIZE=500` (reduce batch size if OOM).
* Add **Top-5 accuracy**: `topk = (logits.topk(5, dim=1).indices == y.unsqueeze(1)).any(1).float().mean()`.

---

## Troubleshooting

* **FileNotFoundError**: check `image_path` strings and that you picked the right size folder (`300p` vs `500p`).
* **MPS/CUDA memory**: lower `batch_size` or image size; close other notebooks.
* **Val F1 is NaN**: ensure `val_df["y"]` has ints (no NaNs), i.e., closed-set labels mapped correctly.

---

This baseline is your anchor. Once it’s stable, we can bolt on **metadata fusion** and measure the Δ in **macro-F1** cleanly.
