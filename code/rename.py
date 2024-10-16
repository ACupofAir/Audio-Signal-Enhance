import os
import re

# 定义一个字典来映射中文到英文的名称
translation_dict = {
    "目标": "Target",
    "第": "No.",
    "个周期": "Cycle",
    "个潜标组": "Group",
    "深度": "Depth"
}

# 定义一个函数来替换中文为英文
def replace_chinese_with_english(chinese_text):
    for chinese, english in translation_dict.items():
        chinese_text = chinese_text.replace(chinese, english)
    return chinese_text

# 遍历文件夹下的所有文件
folder_path = "C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/val/s1"
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        original_name = os.path.splitext(filename)[0]
        english_name = replace_chinese_with_english(original_name)

        # 使用正则表达式提取数字信息
        match = re.search(r'\d+', english_name)
        if match:
            number_info = match.group()
        else:
            number_info = ""

        # 去掉所有分隔符
        english_name = re.sub(r'[^a-zA-Z0-9]+', ' ', english_name).strip()

        # 组合新文件名
        new_filename = f"Data {number_info} {english_name}.wav"

        # 重命名文件
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

print("文件重命名完成")
