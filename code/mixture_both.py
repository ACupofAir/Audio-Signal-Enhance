import os
import random
import soundfile as sf

# 输入音频文件夹路径和输出文件夹路径
# input_folder_s1 = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/train/s1"
# input_folder_s2 = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/train/s2"
# input_folder_noise = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/train/noise"
# output_folder = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/train/mix_both"

input_folder_s1 = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/val/s1"
input_folder_s2 = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/val/s2"
input_folder_noise = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/val/noise"
output_folder = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/val/mix_both"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取s1、s2和noise文件夹中的音频文件列表
s1_files = os.listdir(input_folder_s1)
s2_files = os.listdir(input_folder_s2)
noise_files = os.listdir(input_folder_noise)

# 确保s1、s2和noise文件夹中的音频数量相同
if len(s1_files) != len(s2_files) or len(s1_files) != len(noise_files):
    raise ValueError("s1、s2和noise文件夹中的音频数量不一致")

# 随机选择并处理音频
num_mixes = 630
for _ in range(num_mixes):
    # 随机选择音频文件
    file1 = random.choice(s1_files)
    file2 = random.choice(s2_files)
    noise_file = random.choice(noise_files)

    # 读取音频文件
    audio1, sample_rate1 = sf.read(os.path.join(input_folder_s1, file1))
    audio2, sample_rate2 = sf.read(os.path.join(input_folder_s2, file2))
    noise_audio, noise_sample_rate = sf.read(os.path.join(input_folder_noise, noise_file))

    # 取两个音频和noise中的较短长度作为混合后长度
    min_length = min(len(audio1), len(audio2), len(noise_audio))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]
    noise_audio = noise_audio[:min_length]

    # 执行加性混合
    mixed_audio = audio1 + audio2 + noise_audio

    # Split the filename into the base part and extension part
    base_name, extension = os.path.splitext(file1)

    # Construct the output filename without the extension
    output_file = f"{base_name}_{file2}"

    # 保存混合后的音频
    sf.write(os.path.join(output_folder, output_file), mixed_audio, sample_rate1)

print(f"已生成{num_mixes}条混合音频。")
