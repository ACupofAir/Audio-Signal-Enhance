import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# Function to show magnitude spectrogram with a specific colormap
def show_magspec(waveform, cmap='viridis', **kw):
    return librosa.display.specshow(
        librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max),
        y_axis="log", x_axis="time",
        cmap=cmap,  # 设置颜色映射
        **kw
    )

input_directory = 'C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/test/'
output_directory = 'C:/Users/admin/Desktop/Cola_Software_Build_merge/Asteroid/Sanya/test_pic'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Loop through WAV files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.wav'):
        # Load the WAV file
        filepath = os.path.join(input_directory, filename)
        waveform = sf.read(filepath)[0]

        # Create a plot for the waveform with a specific colormap
        plt.figure(figsize=(7, 5))
        show_magspec(waveform, cmap='viridis', sr=125000)
        plt.colorbar(format="%+2.0f dB")

        # Save the plot in JPEG format with the same name
        output_filename = os.path.splitext(filename)[0] + '.jpeg'
        output_filepath = os.path.join(output_directory, output_filename)
        plt.savefig(output_filepath, format='jpeg')
        plt.close()

print("Plots saved in", output_directory)
