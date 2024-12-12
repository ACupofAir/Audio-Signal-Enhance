import os
import random
import soundfile as sf

# Input and output folder paths
# input_folder_s1 = r"C:\Users\ASUS\junwang\Asteroid\data\train\s1"
# input_folder_s2 = r"C:\Users\ASUS\junwang\Asteroid\data\train\s2"
# output_folder = r"C:\Users\ASUS\junwang\Asteroid\data\train\mix_clean"
input_folder_s1 = r"C:\Users\ASUS\junwang\Asteroid\data\val\s1"
input_folder_s2 = r"C:\Users\ASUS\junwang\Asteroid\data\val\s2"
output_folder = r"C:\Users\ASUS\junwang\Asteroid\data\val\mix_clean"



if not os.path.exists(output_folder):
    os.makedirs(output_folder)

s1_files = os.listdir(input_folder_s1)
s2_files = os.listdir(input_folder_s2)

if len(s1_files) != len(s2_files):
    raise ValueError("s1 and s2 folders contain a different number of audio files")

num_mixes = len(s1_files)
print('======================DEBUG START: num_mixes======================')
print(len(s1_files))
print('======================DEBUG  END : num_mixes======================')

for _ in range(num_mixes):
    file1 = random.choice(s1_files)
    file2 = random.choice(s2_files)

    audio1, sample_rate1 = sf.read(os.path.join(input_folder_s1, file1))
    audio2, sample_rate2 = sf.read(os.path.join(input_folder_s2, file2))

    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]

    mixed_audio = audio1 + audio2

    # Split the filename into the base part and extension part
    base_name, extension = os.path.splitext(file1)

    # Construct the output filename without the extension
    output_file = f"{base_name}_{file2}"

    sf.write(os.path.join(output_folder, output_file), mixed_audio, sample_rate1)

print(f"Generated {num_mixes} mixed audio files.")
