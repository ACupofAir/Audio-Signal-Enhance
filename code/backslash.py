import os
import pandas as pd

metadata_folder = 'C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/metadata'

# 遍历metadata文件夹下的所有CSV文件
for root, dirs, files in os.walk(metadata_folder):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 遍历所有列
            for col in df.columns:
                # 如果列的数据类型是字符串，替换\为/
                if df[col].dtype == 'O':
                    df[col] = df[col].str.replace('\\', '/')

            # 保存修改后的CSV文件
            df.to_csv(file_path, index=False)
