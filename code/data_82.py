import os
import shutil
import random

# 源文件夹
source_folder = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/object1&2_cut"

# 目标文件夹
train_folder = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/train"
val_folder = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/val"

# 创建目标文件夹
for category in ["s1", "s2"]:
    os.makedirs(os.path.join(train_folder, category), exist_ok=True)
    os.makedirs(os.path.join(val_folder, category), exist_ok=True)

# 指定关键字
target1_keyword = "目标1"
target2_keyword = "目标2"

# 获取源文件夹中符合条件的文件列表
target1_files = [f for f in os.listdir(source_folder) if target1_keyword in f]
target2_files = [f for f in os.listdir(source_folder) if target2_keyword in f]

# 定义划分比例
split_ratio = 0.8

# 随机打乱文件列表
random.shuffle(target1_files)
random.shuffle(target2_files)


# 划分文件并复制
def copy_files(source_files, train_folder, val_folder, category):
    split_index = int(len(source_files) * split_ratio)
    train_files = source_files[:split_index]
    val_files = source_files[split_index:]

    for file in train_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(train_folder, category, file)
        shutil.copy(source_path, destination_path)

    for file in val_files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(val_folder, category, file)
        shutil.copy(source_path, destination_path)


# 划分并复制目标1的文件
copy_files(target1_files, train_folder, val_folder, "s1")

# 划分并复制目标2的文件
copy_files(target2_files, train_folder, val_folder, "s2")
