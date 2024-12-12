import os
import numpy as np
import soundfile as sf

# 指定存储随机噪声的目录
train_output_dir = r"C:\Users\ASUS\junwang\Asteroid\data\train\noise"
train_data_num = len(os.listdir(r"C:\Users\ASUS\junwang\Asteroid\data\train\s1"))
os.makedirs(train_output_dir, exist_ok=True)

# 设置采样率和音频长度
sample_rate = 22050  # 125000 Hz
duration = 5

for i in range(0, train_data_num):
    # 生成随机噪声
    random_noise = np.random.normal(0, 1, int(sample_rate * duration))

    # 生成文件名
    filename = os.path.join(train_output_dir, f"noise{i}.wav")

    # 保存为WAV文件
    sf.write(filename, random_noise, sample_rate)

val_output_dir = r"C:\Users\ASUS\junwang\Asteroid\data\val\noise"
val_data_num = len(os.listdir(r"C:\Users\ASUS\junwang\Asteroid\data\val\s1"))
os.makedirs(val_output_dir, exist_ok=True)

# 设置采样率和音频长度
sample_rate = 22050  # 125000 Hz
duration = 5

for i in range(train_data_num, train_data_num+val_data_num):
    # 生成随机噪声
    random_noise = np.random.normal(0, 1, int(sample_rate * duration))

    # 生成文件名
    filename = os.path.join(val_output_dir, f"noise{i}.wav")

    # 保存为WAV文件
    sf.write(filename, random_noise, sample_rate)
print("随机噪声文件生成完成")
