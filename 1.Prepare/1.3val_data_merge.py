import os
import re
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# 定义路径
base_path = r"Type Your Folder"
s1_path = os.path.join(base_path, "Sentinel1_River")
s2_path = os.path.join(base_path, "Sentinel2_River")
output_path = os.path.join(base_path, "Merged")

# 创建输出文件夹
os.makedirs(output_path, exist_ok=True)

# 日志文件
log_file = os.path.join(output_path, "merge_log.txt")
log = open(log_file, "w", encoding="utf-8")
log.write(f"开始处理数据合并: {datetime.now()}\n")

# 波段顺序（去掉 Angle，S1 每季度 4 个波段，共 44 个波段）
band_order = [
    "Q1_VV", "Q1_VH", "Q1_VV-VH", "Q1_VV/VH",
    "Q2_VV", "Q2_VH", "Q2_VV-VH", "Q2_VV/VH",
    "Q3_VV", "Q3_VH", "Q3_VV-VH", "Q3_VV/VH",
    "Q4_VV", "Q4_VH", "Q4_VV-VH", "Q4_VV/VH",
    "Q1_Blue", "Q1_Green", "Q1_Red", "Q1_NIR",
    "Q2_Blue", "Q2_Green", "Q2_Red", "Q2_NIR",
    "Q3_Blue", "Q3_Green", "Q3_Red", "Q3_NIR",
    "Q4_Blue", "Q4_Green", "Q4_Red", "Q4_NIR",
    "Q1_NDVI", "Q1_NDWI", "Q1_WBI",
    "Q2_NDVI", "Q2_NDWI", "Q2_WBI",
    "Q3_NDVI", "Q3_NDWI", "Q3_WBI",
    "Q4_NDVI", "Q4_NDWI", "Q4_WBI"
]

# 解析文件名
def parse_filename(filename):
    match = re.match(r"(S1|S2)_(\d+)_(\d{4})_Q([1-4])\.tif", filename)
    if match:
        sat_type, plot_id, year, quarter = match.groups()
        return sat_type, plot_id, year, int(quarter)
    return None

# 计算指数
def calculate_indices(blue, green, red, nir):
    # 避免除零
    epsilon = 1e-8
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red + epsilon)
    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = (green - nir) / (green + nir + epsilon)
    # WBI = NIR / Red
    wbi = nir / (red + epsilon)
    # 替换 NaN 和 Inf
    ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
    ndwi = np.where(np.isfinite(ndwi), ndwi, 0)
    wbi = np.where(np.isfinite(wbi), wbi, 0)
    return ndvi, ndwi, wbi

# 收集文件
files_s1 = [f for f in os.listdir(s1_path) if f.endswith(".tif")]
files_s2 = [f for f in os.listdir(s2_path) if f.endswith(".tif")]

# 按样方和年份分组
data_groups = defaultdict(lambda: {"S1": {}, "S2": {}})
for file in files_s1:
    parsed = parse_filename(file)
    if parsed:
        sat_type, plot_id, year, quarter = parsed
        key = (plot_id, year)
        data_groups[key]["S1"][quarter] = os.path.join(s1_path, file)
for file in files_s2:
    parsed = parse_filename(file)
    if parsed:
        sat_type, plot_id, year, quarter = parsed
        key = (plot_id, year)
        data_groups[key]["S2"][quarter] = os.path.join(s2_path, file)

# 处理每个样方和年份
for (plot_id, year), data in data_groups.items():
    log.write(f"\n处理样方 {plot_id} 年份 {year}\n")
    s1_files = data["S1"]
    s2_files = data["S2"]

    # 检查是否包含所有季度
    quarters = [1, 2, 3, 4]
    missing_s1 = [q for q in quarters if q not in s1_files]
    missing_s2 = [q for q in quarters if q not in s2_files]
    if missing_s1 or missing_s2:
        log.write(f"警告: 样方 {plot_id} {year} 缺失数据\n")
        if missing_s1:
            log.write(f"  Sentinel-1 缺失季度: {missing_s1}\n")
        if missing_s2:
            log.write(f"  Sentinel-2 缺失季度: {missing_s2}\n")
        continue

    # 读取参考影像以获取元数据
    with rasterio.open(s1_files[1]) as ref:
        profile = ref.profile
        height, width = ref.shape
        crs = ref.crs
        transform = ref.transform

    # 初始化输出数组
    output_bands = np.zeros((len(band_order), height, width), dtype=np.float32)

    # 处理 Sentinel-1 数据（去掉 Angle，4 个波段）
    s1_band_map = {1: 0, 2: 1, 3: 2, 4: 3}  # VV, VH, VV-VH, VV/VH
    for quarter in quarters:
        with rasterio.open(s1_files[quarter]) as src:
            bands = src.read()  # 读取所有波段
            for band_idx, band_name in s1_band_map.items():
                output_idx = (quarter - 1) * 4 + band_idx  # 每季度 4 个波段
                output_bands[output_idx] = bands[band_name]

    # 处理 Sentinel-2 数据并计算指数
    s2_band_map = {1: 0, 2: 1, 3: 2, 4: 3}  # Blue, Green, Red, NIR
    for quarter in quarters:
        with rasterio.open(s2_files[quarter]) as src:
            bands = src.read()
            blue = bands[s2_band_map[1]].astype(np.float32)
            green = bands[s2_band_map[2]].astype(np.float32)
            red = bands[s2_band_map[3]].astype(np.float32)
            nir = bands[s2_band_map[4]].astype(np.float32)

            # 写入原始波段
            base_idx = 16 + (quarter - 1) * 4  # S1 占 16 个波段
            output_bands[base_idx + 0] = blue
            output_bands[base_idx + 1] = green
            output_bands[base_idx + 2] = red
            output_bands[base_idx + 3] = nir

            # 计算指数
            ndvi, ndwi, wbi = calculate_indices(blue, green, red, nir)
            idx_base = 32 + (quarter - 1) * 3  # S1 16 + S2 16 = 32
            output_bands[idx_base + 0] = ndvi
            output_bands[idx_base + 1] = ndwi
            output_bands[idx_base + 2] = wbi

    # 更新 profile
    profile.update(
        count=len(band_order),
        dtype=rasterio.float32,
        compress="lzw"
    )

    # 写入输出文件
    output_file = os.path.join(output_path, f"Merged_{plot_id}_{year}.tif")
    with rasterio.open(output_file, "w", **profile) as dst:
        for i, band_name in enumerate(band_order, 1):
            dst.write(output_bands[i - 1], i)
            dst.set_band_description(i, band_name)
    
    log.write(f"成功生成: {output_file}\n")

log.write(f"\n处理完成: {datetime.now()}\n")
log.close()

print(f"数据合并完成，输出文件位于 {output_path}")
print(f"日志文件: {log_file}")