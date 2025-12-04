import os
import re
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from rasterio.enums import Resampling

# 配置路径
input_dir = r'Type Your Folder'
output_dir = r'Type Your Folder'
os.makedirs(output_dir, exist_ok=True)

# 波段命名规则
BAND_NAMES = {
    'S1': ['VV', 'VH', 'VV-VH', 'VV/VH'],
    'S2': ['B', 'G', 'R', 'NIR']
}

def parse_filename(filename):
    """解析新格式文件名获取元数据"""
    # 匹配S2格式（示例：2024-S2-Q1-0000014848-0000029696.tif）
    s2_match = re.match(r'^2024-S2-Q(\d)-(\d+)-(\d+)\.tif', filename)
    if s2_match:
        return {
            'year': '2024',
            'quarter': f'Q{s2_match.group(1)}',
            'satellite': 'S2',
            'tile_x': int(s2_match.group(2)),
            'tile_y': int(s2_match.group(3)),
            'sort_key': (int(s2_match.group(2)), int(s2_match.group(3)))
        }
    
    # 匹配S1格式（示例：2024_Q4-0000014848-0000029696.tif）
    s1_match = re.match(r'^2024_Q(\d)-(\d+)-(\d+)\.tif', filename)
    if s1_match:
        return {
            'year': '2024',
            'quarter': f'Q{s1_match.group(1)}',
            'satellite': 'S1',
            'tile_x': int(s1_match.group(2)),
            'tile_y': int(s1_match.group(3)),
            'sort_key': (int(s1_match.group(2)), int(s1_match.group(3)))
        }
    return None

def merge_tiles(file_list):
    """空间拼接分块数据"""
    src_files = [rasterio.open(f) for f in file_list]
    mosaic, out_trans = merge(src_files)
    
    # 创建输出元数据
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0],
        "crs": src_files[0].crs
    })
    return mosaic, out_meta

def process_files():
    # 1. 读取并分类文件
    all_files = glob.glob(os.path.join(input_dir, '*.tif'))
    if not all_files:
        print("错误：未找到任何TIFF文件！请检查：")
        print("1. 输入路径是否正确？")
        print("2. 文件扩展名是否为.tif（注意大小写）？")
        print("3. 是否有隐藏字符？")
        return
    data_catalog = {
        'S1': {q: [] for q in ['Q1','Q2','Q3','Q4']},
        'S2': {q: [] for q in ['Q1','Q2','Q3','Q4']}
    }

    for fp in all_files:
        meta = parse_filename(os.path.basename(fp))
        if not meta:
            continue
        data_catalog[meta['satellite']][meta['quarter']].append(fp)
    print("Strating Step 2")
    # 2. 合并分块数据
    merged_data = []
    for sat in ['S1', 'S2']:
        for q in ['Q1','Q2','Q3','Q4']:
            files = data_catalog[sat][q]
            if len(files) < 1:
                continue  # 无数据时跳过
                
            # 按tile_x和tile_y排序（确保空间连续性）
            files.sort(key=lambda x: x[1])
            file_paths = [f[0] for f in files]
            
            # 执行合并
            mosaic, meta = merge_tiles(file_paths)
            
            # 保存临时合并文件
            temp_path = os.path.join(output_dir, f"temp_{sat}_{q}.tif")
            with rasterio.open(temp_path, 'w', **meta) as dst:
                dst.write(mosaic)
            merged_data.append((sat, q, temp_path))
    print(merged_data)
    print("Strating Step 3")
    # 3. 按时间顺序排列
    sorted_data = []
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    for q in quarters:
        s1_file = next((item[2] for item in merged_data 
                       if item[0]=='S1' and item[1]==q), None)
        s2_file = next((item[2] for item in merged_data 
                       if item[0]=='S2' and item[1]==q), None)
        if s1_file:
            sorted_data.append(('S1', q, s1_file))
        if s2_file:
            sorted_data.append(('S2', q, s2_file))

    # 4. 波段重命名并保存最终结果
    for sat, q, temp_path in sorted_data:
        with rasterio.open(temp_path) as src:
            output_path = os.path.join(output_dir, 
                f"2024-{q}-{sat}_merged.tif")
            
            meta = src.meta.copy()
            new_descriptions = BAND_NAMES[sat]
            
            if src.count != len(new_descriptions):
                print(f"错误：{sat}-{q} 波段数量不匹配，跳过")
                continue
                
            meta.update({
                'descriptions': new_descriptions,
                'driver': 'GTiff'
            })
 
            with rasterio.open(output_path, 'w', **meta) as dst:
                for i in range(1, src.count+1):
                    band_data = src.read(i)
                    dst.write(band_data, i)
                    dst.set_band_description(i, new_descriptions[i-1])
 
        os.remove(temp_path)
if __name__ == '__main__':
    process_files()
    print("处理完成！结果保存在：", output_dir)