import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score, recall_score
import rasterio
from PIL import Image
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2  # Added for save_predictions

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

if __name__ == '__main__':
    image_dir = r"Type your image folder"
    label_dir = r"Type your label folder"
    image_dir_processed = r"Type your image_processed folder"
    label_dir_processed = r"Type your label_processed folder"
    train_output_dir = r"Type your train_output folder"
    pred_output_dir = r"Type your pred_output folder"
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(pred_output_dir, exist_ok=True)

    sample_ids = preprocess_images_and_labels(image_dir, label_dir, image_dir_processed, label_dir_processed)
    train_ids, pred_ids = train_test_split(sample_ids, test_size=0.2, random_state=42)

    train_dataset = ImageDataset(image_dir_processed, label_dir_processed, train_ids)
    pred_dataset = ImageDataset(image_dir_processed, label_dir_processed, pred_ids)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetCnnSwinImage(in_channels=44).to(device)

    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    model.train()
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            y_batch_resized = F.interpolate(y_batch, size=outputs.shape[2:], mode='bilinear', align_corners=False)
            loss_bce = criterion_bce(outputs, y_batch_resized)
            loss_dice = criterion_dice(outputs, y_batch_resized)
            loss = 0.5 * loss_bce + 0.5 * loss_dice
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

    save_predictions(train_loader, train_output_dir, train_ids, prefix="train")
    save_predictions(pred_loader, pred_output_dir, pred_ids, prefix="pred")