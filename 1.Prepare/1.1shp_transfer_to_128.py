import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import uuid

def expand_rectangle(geometry, orig_cols=63, orig_rows=51, target_size=130):
   
    # 获取矩形中心点
    centroid = geometry.centroid
    cx, cy = centroid.x, centroid.y
    
    # 获取原始矩形的边界
    minx, miny, maxx, maxy = geometry.bounds
    orig_width = maxx - minx
    orig_height = maxy - miny
    
    # 计算缩放比例
    scale_x = target_size / orig_cols
    scale_y = target_size / orig_rows
    
    # 计算新的宽和高
    new_width = orig_width * scale_x
    new_height = orig_height * scale_y
    
    # 计算新的矩形半宽和半高
    half_width = new_width / 2
    half_height = new_height / 2
    
    # 创建新的矩形坐标
    new_coords = [
        (cx - half_width, cy - half_height),
        (cx + half_width, cy - half_height),
        (cx + half_width, cy + half_height),
        (cx - half_width, cy + half_height),
        (cx - half_width, cy - half_height)  # 闭合多边形
    ]
    
    return Polygon(new_coords)

def process_shp(input_shp, output_shp, target_size=130):
    """
    处理SHP文件：按比例扩大矩形并添加编号
    """
    try:
        # 读取SHP文件
        gdf = gpd.read_file(input_shp)
        
        # 验证输入数据
        if len(gdf) != 800:
            print(f"警告：SHP文件中包含 {len(gdf)} 个要素，而不是预期的800个")
        
        # 扩大每个矩形
        gdf['geometry'] = gdf['geometry'].apply(
            lambda geom: expand_rectangle(geom, orig_cols=63, orig_rows=51, target_size=target_size)
        )
        
        # 添加编号列（1-800）
        gdf['ID'] = range(1, len(gdf) + 1)
        
        # 保存新的SHP文件
        gdf.to_file(output_shp, encoding='utf-8')
        print(f"成功处理并保存到 {output_shp}")
        
        return gdf
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None

def main():
    # 输入和输出文件路径
    input_shp = r"Type Your SHP"
    output_shp = r"Type Your SHP"
    
    # 处理SHP文件
    result_gdf = process_shp(input_shp, output_shp)
    
    if result_gdf is not None:
        print(f"处理完成，共处理 {len(result_gdf)} 个样方")
        print("新SHP文件包含以下列：")
        print(result_gdf.columns)

if __name__ == "__main__":
    main()

