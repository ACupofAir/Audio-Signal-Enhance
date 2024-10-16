import torch
from asteroid.models import DPRNNTasNet
from asteroid.losses import PITLossWrapper
from asteroid.losses.sdr import PairwiseNegSDR
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save_path = "DPRNN-bs2_epoch30_sr125000.pth"
audio_input_path = r"C:\Users\june\Workspace\Asteroid\Sanya\code\input_test.wav"

# Load the model state dict
model = torch.load(model_save_path)

# Ensure the model is in evaluation mode
model.eval()

# Perform separation on new data
model.separate(
    audio_input_path,
    resample=True,
)


audio_output_path = r"C:\Users\june\Workspace\Asteroid\Sanya\code\input_test_est1.wav"
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
print(loss)
