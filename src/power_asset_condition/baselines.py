"""Representative baselines trained on the published full training partition."""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def hog_descriptor(image_path: str) -> np.ndarray:
    """128px, 8px cells, 9 unsigned orientation bins, 2x2 blocks."""
    image = np.asarray(Image.open(image_path).convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    gy, gx = np.gradient(image)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.degrees(np.arctan2(gy, gx)), 180.0)
    position = angle * (9.0 / 180.0)
    lower = np.floor(position).astype(np.int32) % 9
    fraction = position - np.floor(position)
    upper = (lower + 1) % 9
    hist = np.zeros((16, 16, 9), dtype=np.float32)
    cell = np.broadcast_to(np.arange(128)[:, None] // 8 * 16 + np.arange(128)[None, :] // 8,
                           (128, 128)).ravel()
    np.add.at(hist.reshape(-1, 9), (cell, lower.ravel()), (magnitude * (1 - fraction)).ravel())
    np.add.at(hist.reshape(-1, 9), (cell, upper.ravel()), (magnitude * fraction).ravel())
    blocks = []
    for y in range(15):
        for x in range(15):
            block = hist[y:y + 2, x:x + 2].reshape(-1)
            block /= np.sqrt(np.dot(block, block) + 1e-8)
            block = np.minimum(block, 0.2)
            block /= np.sqrt(np.dot(block, block) + 1e-8)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def train_hog_svm(fit_rows: list[dict], query_rows: list[dict], output_dir: Path, seed: int = 20261301) -> dict:
    x = np.stack([hog_descriptor(r["image_path"]) for r in fit_rows])
    q = np.stack([hog_descriptor(r["image_path"]) for r in query_rows])
    y = np.array([r["leaf_id"] for r in fit_rows])
    groups = np.array([r["group_id"] for r in fit_rows])
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    candidates = [(c, gamma) for c in (0.01, 0.1, 1, 10, 100) for gamma in ("scale", 1e-5, 1e-4, 1e-3)]
    best = None
    for c, gamma in candidates:
        folds = []
        for tr, va in splitter.split(x, y, groups):
            model = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, kernel="rbf", probability=True, random_state=seed))
            model.fit(x[tr], y[tr])
            folds.append(float((model.predict(x[va]) == y[va]).mean()))
        score = float(np.mean(folds))
        if best is None or score > best[0]:
            best = (score, c, gamma)
    _, c, gamma = best
    model = make_pipeline(StandardScaler(), SVC(C=c, gamma=gamma, kernel="rbf", probability=True, random_state=seed))
    model.fit(x, y)
    probabilities = model.predict_proba(q)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "predictions.npz",
                        sample_id=np.array([r["sample_id"] for r in query_rows]), probabilities=probabilities)
    return {"method": "hog_rbf_svm", "selected_c": c, "selected_gamma": gamma,
            "inner_accuracy": best[0], "fit_rows": len(fit_rows), "query_rows": len(query_rows)}


def train_convnext(fit_rows: list[dict], query_rows: list[dict], output_dir: Path,
                   *, seed: int = 20261301, epochs: int = 6, pretrained: bool = True) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.convnext_tiny(weights=weights)
    in_features = net.classifier[2].in_features
    net.classifier[2] = torch.nn.Linear(in_features, 11)
    net.to("cuda:0")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(192, scale=(0.75, 1.0), ratio=(0.75, 4 / 3)),
        transforms.RandomHorizontalFlip(0.5), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(236), transforms.CenterCrop(192), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    class Images(Dataset):
        def __init__(self, rows, transform):
            self.rows, self.transform = rows, transform
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            with Image.open(row["image_path"]) as im:
                image = self.transform(im.convert("RGB"))
            return image, int(row.get("leaf_id", -1))
    loader = DataLoader(Images(fit_rows, train_transform), batch_size=16, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW([
        {"params": [p for n, p in net.named_parameters() if not n.startswith("classifier.2")], "lr": 1e-4},
        {"params": net.classifier[2].parameters(), "lr": 1e-3}], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    for _ in range(epochs):
        net.train(); opt.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(loader):
            logits = net(images.to("cuda:0"))
            loss = torch.nn.functional.cross_entropy(logits, labels.to("cuda:0")) / 4
            loss.backward()
            if (step + 1) % 4 == 0 or step + 1 == len(loader):
                opt.step(); opt.zero_grad(set_to_none=True)
        scheduler.step()
    net.eval()
    query_loader = DataLoader(Images(query_rows, eval_transform), batch_size=32, shuffle=False, num_workers=2)
    parts = []
    with torch.inference_mode():
        for images, _ in query_loader:
            parts.append(torch.softmax(net(images.to("cuda:0")), dim=-1).cpu().numpy())
    probabilities = np.concatenate(parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), output_dir / "convnext_tiny.pt")
    np.savez_compressed(output_dir / "predictions.npz", sample_id=np.array([r["sample_id"] for r in query_rows]),
                        probabilities=probabilities)
    return {"method": "convnext_tiny", "epochs": epochs, "pretrained": pretrained,
            "fit_rows": len(fit_rows), "query_rows": len(query_rows)}


def train_yolo(fit_rows: list[dict], query_rows: list[dict], output_dir: Path,
               *, checkpoint: str, seed: int = 20261301, epochs: int = 8) -> dict:
    from ultralytics import YOLO

    dataset = output_dir.resolve() / "yolo_dataset"
    for label in range(11):
        (dataset / "train" / f"{label:02d}").mkdir(parents=True, exist_ok=True)
    for row in fit_rows:
        source = Path(row["image_path"]).resolve(strict=True)
        link = dataset / "train" / f"{row['leaf_id']:02d}" / f"{row['sample_id']}{source.suffix.lower()}"
        if not link.exists():
            link.symlink_to(source)
    val_alias = dataset / "val"
    if val_alias.is_symlink():
        val_alias.unlink()
    if not val_alias.exists():
        val_alias.symlink_to("train", target_is_directory=True)
    model = YOLO(checkpoint)
    model.train(data=str(dataset), epochs=epochs, imgsz=224, batch=64, workers=4, device=0,
                project=str(output_dir / "ultralytics"), name="train", exist_ok=True,
                optimizer="AdamW", lr0=1e-4, lrf=0.01, weight_decay=0.05,
                cos_lr=True, warmup_epochs=0, patience=0, val=False, save=True,
                plots=False, seed=seed, deterministic=True, cache=False,
                auto_augment=None, erasing=0.0, fliplr=0.5, flipud=0.0,
                mosaic=0.0, mixup=0.0)
    last = output_dir / "ultralytics/train/weights/last.pt"
    model = YOLO(str(last))
    probabilities = np.stack([
        np.asarray(result.probs.data.cpu(), dtype=np.float32)
        for row in query_rows
        for result in model.predict(source=row["image_path"], imgsz=224, device=0, verbose=False)
    ])
    np.savez_compressed(output_dir / "predictions.npz", sample_id=np.array([r["sample_id"] for r in query_rows]),
                        probabilities=probabilities)
    return {"method": "yolov8n_cls", "epochs": epochs, "fit_rows": len(fit_rows), "query_rows": len(query_rows)}
