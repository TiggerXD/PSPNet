import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
import multiprocessing
import importlib
import warnings

HERE = os.path.dirname(__file__) or "."
PSPNET_CANDIDATES = [
    os.path.join(HERE, "pspnet-pytorch"),
    os.path.join(HERE, "pspnet_pytorch"),
    os.path.join(HERE, "pspnet"),
    os.path.join(HERE, "pspnet_pytorch-master"),
]

SPLITS_DIR = os.path.join(HERE, "splits")
TRAIN_JSON = os.path.join(SPLITS_DIR, "train.json")
VAL_JSON   = os.path.join(SPLITS_DIR, "val.json")
TEST_JSON  = os.path.join(SPLITS_DIR, "test.json")
IMG_DIR    = os.path.join(HERE, "train", "images")

BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
SAVE_PATH = os.path.join(HERE, "pspnet_model.pth")

TARGET_SIZE = (512, 512)

pspnet_folder = None
for c in PSPNET_CANDIDATES:
    if os.path.isdir(c):
        pspnet_folder = c
        break

if pspnet_folder is None:
    for name in os.listdir(HERE):
        path = os.path.join(HERE, name)
        if os.path.isdir(path) and "pspnet.py" in os.listdir(path):
            pspnet_folder = path
            break

if pspnet_folder is None:
    raise FileNotFoundError(f"pspnet folder not found. Searched: {PSPNET_CANDIDATES} and top-level folders in {HERE}")

sys.path.insert(0, pspnet_folder)
print("DEBUG: HERE =", HERE)
print("DEBUG: PSPNET_FOLDER =", pspnet_folder)
print("DEBUG: sys.path[0] =", sys.path[0])
print("DEBUG: files in pspnet folder:", os.listdir(pspnet_folder))

pspnet = importlib.import_module("pspnet")
PSPNet = getattr(pspnet, "PSPNet")

class COCOSegmentationDataset(Dataset):
    def __init__(self, json_file, img_dir):
        self.coco = COCO(json_file)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        self.num_classes = len(cat_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        img_path = os.path.join(self.img_dir, path)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)

        for ann in anns:
            try:
                segm = ann.get("segmentation", None)
                if segm is None:
                    continue

                if ann.get("iscrowd", 0) == 1:
                    if isinstance(segm, list):
                        rle = maskUtils.frPyObjects(segm, img_info["height"], img_info["width"])
                    else:
                        rle = segm
                    seg = maskUtils.decode(rle)
                else:
                    if isinstance(segm, list) and len(segm) == 0:
                        continue
                    rles = maskUtils.frPyObjects(segm, img_info["height"], img_info["width"])
                    seg = maskUtils.decode(rles)

                if seg.ndim == 3:
                    seg = np.any(seg, axis=2).astype(np.uint8)

                cat_id = ann.get("category_id", None)
                if cat_id is None:
                    continue
                if cat_id not in self.cat_id_to_index:
                    continue
                label = self.cat_id_to_index[cat_id]

                mask[seg == 1] = label

            except Exception as e:
                print(f"WARNING: skipping ann id {ann.get('id')} for img {img_id} due to error: {e}")
                continue

        target_h, target_w = TARGET_SIZE
        img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        img_tensor = transforms.ToTensor()(img_resized)
        mask_tensor = torch.from_numpy(mask_resized).long()

        return img_tensor, mask_tensor

def extract_logits(model_output):
    if isinstance(model_output, dict):
        return model_output.get("out", list(model_output.values())[0])
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output

def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            logits = extract_logits(outputs)
            if logits.shape[2:] != masks.shape[1:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[1:], mode="bilinear", align_corners=False
                )
            loss = criterion(logits, masks)
            total_loss += loss.item()
    avg_loss = total_loss / max(1, len(loader))
    print(f"{split_name} Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    for path in [TRAIN_JSON, VAL_JSON, TEST_JSON]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON not found: {path}")
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMG_DIR}")

    train_ds = COCOSegmentationDataset(TRAIN_JSON, IMG_DIR)
    val_ds   = COCOSegmentationDataset(VAL_JSON, IMG_DIR)
    test_ds  = COCOSegmentationDataset(TEST_JSON, IMG_DIR)
    num_classes = train_ds.num_classes
    print(f"Detected {num_classes} classes")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = PSPNet(n_classes=num_classes, pretrained=False, backend="resnet34")
    print("Instantiated PSPNet with backend=resnet34, pretrained=False")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{NUM_EPOCHS}", ncols=100)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            logits = extract_logits(outputs)
            if logits.shape[2:] != masks.shape[1:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[1:], mode="bilinear", align_corners=False
                )
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix_str(f"loss={loss.item():.4f}")

        avg_train_loss = train_loss / max(1, len(train_loader))
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f}")

        evaluate(model, val_loader, criterion, device, split_name="Val")

        torch.save(model.state_dict(), SAVE_PATH)

    print("Final evaluation on test set:")
    evaluate(model, test_loader, criterion, device, split_name="Test")

    print("Training complete. Model saved to", SAVE_PATH)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
