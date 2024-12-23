# %%
import torch
from asteroid.losses import PITLossWrapper
from asteroid.losses.sdr import PairwiseNegSDR
import torchaudio
import os

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "../checkpoint/DPRNN-bs2_epoch30_sr125000.pth"
audio_input_path = "./example.wav"

# Load the model state dict
model = torch.load(model_save_path)

# Ensure the model is in evaluation mode
model.eval()

# %%
# Perform separation on new data
output_folder = "output"
model.separate(
    audio_input_path,
    output_dir=output_folder,
    resample=False,
)


# %%
# Construct the output audio path
audio_output_path = os.path.join(
    output_folder, os.path.basename(audio_input_path).replace(".wav", "_est1.wav")
)
# %%
# Load the input and output audio files
audio_input, sr_input = torchaudio.load(audio_input_path)
audio_input = audio_input.unsqueeze(0)
print(sr_input, audio_input.shape)
audio_output, sr_output = torchaudio.load(audio_output_path)
audio_output = audio_output.unsqueeze(0)

# Ensure the audio tensors are on the same device as the model
audio_input = audio_input.to(device)
audio_output = audio_output.to(device)
loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx").to(device)
loss = loss_func(audio_output, audio_input)
print(f"{-1* loss.item():.2f} dB")

# %%
