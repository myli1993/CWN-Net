def preprocess_images_and_labels(image_dir, label_dir, output_image_dir, output_label_dir, target_size=(128, 128)):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    sample_ids = list(range(1, 800))
    invalid_samples = []
    for img_id in sample_ids:
        img_path = os.path.join(image_dir, f"ID{img_id}_2024.tif")
        label_path = os.path.join(label_dir, f"{img_id}.png")
        output_img_path = os.path.join(output_image_dir, f"ID{img_id}_2024.tif")
        output_label_path = os.path.join(output_label_dir, f"{img_id}.png")
        try:
            with rasterio.open(img_path) as src:
                img = src.read()
                if img.shape[0] != 32:
                    invalid_samples.append(img_id)
                    continue
                img = np.transpose(img, (1, 2, 0))
                h, w = img.shape[:2]
                result = np.zeros((target_size[0], target_size[1], img.shape[2]), dtype=img.dtype)
                start_h = max(0, (h - target_size[0]) // 2)
                start_w = max(0, (w - target_size[1]) // 2)
                end_h = min(h, start_h + target_size[0])
                end_w = min(w, start_w + target_size[1])
                target_start_h = max(0, (target_size[0] - h) // 2)
                target_start_w = max(0, (target_size[1] - w) // 2)
                target_end_h = target_start_h + (end_h - start_h)
                target_end_w = target_start_w + (end_w - start_w)
                result[target_start_h:target_end_h, target_start_w:target_end_w, :] = img[start_h:end_h, start_w:end_w, :]
                with rasterio.open(output_img_path, 'w', driver='GTiff',
                                  height=target_size[0], width=target_size[1], count=img.shape[2],
                                  dtype=img.dtype, crs=src.crs, transform=src.transform) as dst:
                    dst.write(np.transpose(result, (2, 0, 1)))
        except:
            invalid_samples.append(img_id)
            continue
        try:
            label = np.array(Image.open(label_path))
            h, w = label.shape
            result = np.zeros(target_size, dtype=label.dtype)
            start_h = max(0, (h - target_size[0]) // 2)
            start_w = max(0, (w - target_size[1]) // 2)
            end_h = min(h, start_h + target_size[0])
            end_w = min(w, start_w + target_size[1])
            target_start_h = max(0, (target_size[0] - h) // 2)
            target_start_w = max(0, (target_size[1] - w) // 2)
            target_end_h = target_start_h + (end_h - start_h)
            target_end_w = target_start_w + (end_w - start_w)
            result[target_start_h:target_end_h, target_start_w:target_end_w] = label[start_h:end_h, start_w:end_w]
            Image.fromarray(result).save(output_label_path)
        except:
            invalid_samples.append(img_id)
            continue
    return [id for id in sample_ids if id not in invalid_samples]