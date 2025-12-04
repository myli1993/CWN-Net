def save_predictions(loader, output_dir, sample_ids, prefix="train"):
    model.eval()
    with torch.no_grad():
        for i, (X_batch, _) in enumerate(loader):
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu()
            preds = F.interpolate(preds, size=(128, 128), mode='bilinear', align_corners=False)
            preds = (preds.numpy() > 0.5).astype(np.uint8)
            for j in range(preds.shape[0]):
                pred = preds[j, 0]
                # 形态学操作：开运算
                kernel = np.ones((3, 3), np.uint8)
                pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel) * 255
                sample_idx = sample_ids[i * loader.batch_size + j]
                pred_img = Image.fromarray(pred)
                pred_img.save(os.path.join(output_dir, f"{prefix}_{sample_idx}.png"))