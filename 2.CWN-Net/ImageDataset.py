class ImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, sample_ids, expected_channels=32):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.sample_ids = sample_ids
        self.target_size = (128, 128)
        self.expected_channels = expected_channels
        self.band_names = [
            'Q1_VV', 'Q1_VH', 'Q1_VV-VH', 'Q1_VV/VH',
            'Q2_VV', 'Q2_VH', 'Q2_VV-VH', 'Q2_VV/VH',
            'Q3_VV', 'Q3_VH', 'Q3_VV-VH', 'Q3_VV/VH',
            'Q4_VV', 'Q4_VH', 'Q4_VV-VH', 'Q4_VV/VH',
            'Q1_Blue', 'Q1_Green', 'Q1_Red', 'Q1_NIR',
            'Q2_Blue', 'Q2_Green', 'Q2_Red', 'Q2_NIR',
            'Q3_Blue', 'Q3_Green', 'Q3_Red', 'Q3_NIR',
            'Q4_Blue', 'Q4_Green', 'Q4_Red', 'Q4_NIR'
        ]
        self.band_indices = {name: idx for idx, name in enumerate(self.band_names)}

    def __len__(self):
        return len(self.sample_ids)

    def compute_vegetation_indices(self, img, img_id):
        indices = []
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            nir = np.clip(img[:, :, self.band_indices[f'{quarter}_NIR']], 0, None)
            red = np.clip(img[:, :, self.band_indices[f'{quarter}_Red']], 0, None)
            green = np.clip(img[:, :, self.band_indices[f'{quarter}_Green']], 0, None)
            ndvi = (nir - red) / (nir + red + 1e-8)  # Small epsilon to prevent division by zero
            ndwi = (green - nir) / (green + nir + 1e-8)
            wbi = nir / (red + 1e-8)
            ndvi = np.clip(ndvi, -1, 1)
            ndwi = np.clip(ndwi, -1, 1)
            wbi = np.clip(wbi, 0, 100)
            indices.extend([ndvi, ndwi, wbi])
        indices = np.stack(indices, axis=-1)
        # Replace NaN or Inf with 0
        indices = np.nan_to_num(indices, nan=0.0, posinf=0.0, neginf=0.0)
        return indices

    def __getitem__(self, idx):
        img_id = self.sample_ids[idx]
        img_path = os.path.join(self.image_dir, f"ID{img_id}_2024.tif")
        label_path = os.path.join(self.label_dir, f"{img_id}.png")
        try:
            with rasterio.open(img_path) as src:
                img = src.read()
                if img.shape[0] != self.expected_channels:
                    raise ValueError(f"样本 {img_id} 包含 {img.shape[0]} 个波段，期望 {self.expected_channels}")
                img = np.transpose(img, (1, 2, 0))
            indices = self.compute_vegetation_indices(img, img_id)
            img = np.concatenate([img, indices], axis=-1)
            label = np.array(Image.open(label_path))
            label = (label > 0).astype(np.float32)
            img = torch.FloatTensor(img).permute(2, 0, 1)
            label = torch.FloatTensor(label).unsqueeze(0)
            return img, label
        except Exception as e:
            print(f"Error processing sample {img_id}: {str(e)}, skipping...")
            return self.__getitem__((idx + 1) % len(self.sample_ids))  # Move to next sample