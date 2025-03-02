import openslide
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage import color, filters
import numpy as np

# 打开svs文件
slide_path = './dataset/TCGA-A2-A0T4-01A-03-TSC.60866100-7f22-4c86-a99e-5e52ca53c313.svs'
slide = openslide.OpenSlide(slide_path)

# 读取图像基本信息
level_count = slide.level_count
dimensions = slide.dimensions
properties = slide.properties
print('层级数量：{}，\n各层级分辨率：{}\n放大倍数：{}'.format(level_count, dimensions, properties["aperio.AppMag"]))
print("元数据：")
for key, value in properties.items():
    print(f"{key}: {value}")

# 获取全切片缩略图2000x2000
thumbnail = slide.get_thumbnail((2000, 2000))
plt.figure(figsize=(10, 8))
plt.imshow(thumbnail)
plt.axis('off')
plt.title("WSI_Thumbnail")
plt.show()

# 局部细节展示(按层级显示高分辨率区域)
level = 0  # 选用高分辨率层级
width, height = 1024, 1024  # 提取块的大小
x, y = 50000, 10000  # 指定读取块儿的起始位置

region = slide.read_region((x, y), level, (width, height))
plt.imshow(region)
plt.axis('off')
plt.title("High-Resolution Region (Level_{},size_{}x{})".format(level, width, height))
plt.show()


# 图像切片 patch_size=1024x1024 并过滤空白patch
# 记录切片数
patches_number = 0


def extract_patches(slide, patch_size, output_dir, area_ratio=0.1, level=0):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    width, height = slide.level_dimensions[level]
    # 切片
    patch_height, patch_width = patch_size
    for y in tqdm(range(0, height, patch_height), desc="Processing rows"):  # 显示进度条
        for x in range(0, width, patch_width):
            patch = slide.read_region((x, y), level, (min(patch_width, width - x), min(patch_height, height - y)))
            patch = patch.convert('RGB')

            # 过滤空白patch
            gray = color.rgb2gray(np.array(patch))
            threshold = 220/255.0  # 选择固定阈值
            mask = gray > threshold
            white_area = np.mean(mask)
            if white_area < (1 - area_ratio):
                global patches_number
                patches_number += 1
                # 保存到指定文件夹，格式PNG
                patch.save(os.path.join(output_dir,"level{}_x{}_y{}.png".format(level, x, y)))


# 调用函数进行切片及过滤
patches_size=(1024,1024)
output_dir="./dataset/patches"
extract_patches(slide, patches_size, output_dir)
print("Successfully extracted patches! patches_number:{}".format(patches_number))
