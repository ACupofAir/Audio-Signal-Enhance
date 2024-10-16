import os
import numpy as np
import soundfile as sf

# 指定存储随机噪声的目录
output_dir = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/val/noise"  # 确保目录存在
os.makedirs(output_dir, exist_ok=True)

# 设置采样率和音频长度
sample_rate = 125000  # 125000 Hz
duration = 0.6

# 生成3150条随机噪声并保存为WAV文件
for i in range(2521, 3151):
    # 生成随机噪声
    random_noise = np.random.normal(0, 1, int(sample_rate * duration))

    # 生成文件名
    filename = os.path.join(output_dir, f"noise{i}.wav")

    # 保存为WAV文件
    sf.write(filename, random_noise, sample_rate)

print("随机噪声文件生成完成")
