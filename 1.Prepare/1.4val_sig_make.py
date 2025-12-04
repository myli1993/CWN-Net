import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import geometry_mask
from rasterio.windows import Window
import os
from pathlib import Path

# 设置文件路径
shp_path = r"E:\水科院工作汇总\个人成果\论文\聊城德州\data\0.shp\River_Classify_Merge_Expanded2.shp"
raster_path = r"E:\水科院工作汇总\个人成果\论文\聊城德州\GIS\GIS\Feature_Val_1.tif"
output_dir = r"E:\水科院工作汇总\个人成果\论文\聊城德州\data\1.model\label_new"  # 输出目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取shapefile
gdf = gpd.read_file(shp_path)

# 打开栅格文件
raster = rasterio.open(raster_path)

# 遍历每个样方
for idx, row in gdf.iterrows():
    sample_id = str(row['ID'])
    geometry = row.geometry
    
    # 获取样方的边界框
    minx, miny, maxx, maxy = geometry.bounds
    
    # 创建窗口
    window = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=raster.transform)
    
    # 读取窗口数据
    data = raster.read(1, window=window)
    
    # 二值化处理：0为非水网（黑色），1为水网（白色）
    binary_data = (data != 0).astype(np.uint8)
    
    # 检查是否全为非水网（全0）
    if np.all(binary_data == 0):
        continue
    
    # 创建掩膜
    mask = geometry_mask([geometry], transform=raster.window_transform(window), 
                        out_shape=binary_data.shape, invert=True)
    
    # 应用掩膜
    sample_data = binary_data * mask
    
    # 如果掩膜后仍全为0，跳过
    if np.all(sample_data == 0):
        continue
    
    # 更新元数据用于输出
    out_meta = raster.meta.copy()
    out_meta.update({
        "height": sample_data.shape[0],
        "width": sample_data.shape[1],
        "transform": raster.window_transform(window),
        "driver": "PNG",
        "dtype": "uint8",
        "count": 1
    })
    
    # 保存为PNG文件
    output_path = os.path.join(output_dir, f"{sample_id}.png")
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        # 确保0为黑色，1为白色
        dst.write(sample_data * 255, 1)  # 乘以255使1显示为白色

# 关闭栅格文件
raster.close()

print("处理完成！输出的PNG文件保存在：", output_dir)