import os
from pydub import AudioSegment
# 保留前0.09s，并去除0.047s到0.064s之间的部分

# 输入文件夹路径
input_folder = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/object1&2"

# 输出文件夹路径
output_folder = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/object1&2_cut"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)

        # 读取音频文件
        audio = AudioSegment.from_file(input_path)

        # 裁剪前0.09秒的部分
        audio = audio[:90]  # 0.09秒等于90毫秒

        # 指定裁剪的时间范围（毫秒）
        start_time = 47  # 0.047秒
        end_time = 64  # 0.064秒

        # 裁剪指定时间范围的部分
        part_before = audio[:start_time]
        part_after = audio[end_time:]

        # 拼接部分
        final_audio = part_before + part_after

        # 构建输出文件路径
        output_path = os.path.join(output_folder, filename)

        # 保存裁剪和拼接后的音频
        final_audio.export(output_path, format="wav")
