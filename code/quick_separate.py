from asteroid.models import BaseModel
import soundfile as sf
import torch

# 'from_pretrained' automatically uses the right model class (asteroid.models.DPRNNTasNet).
model = BaseModel.from_pretrained("mpariente/DPRNNTasNet-ks2_WHAM_sepclean")

# model = torch.load('DPRNNTasNet-bs16_epoch20_sr8000.pth')
# model = torch.load('DPRNNTasNet-ks2_Libri1Mix_enhsingle_16k.pth')

# You can pass a NumPy array:
mixture, _ = sf.read("Data 1 Target1No 1Cycle No 1Group Depth500m_Data 2 Target2No 10Cycle No 6Group Depth1200m.wav", dtype = "float32", always_2d=True)
# Soundfile returns the mixture as shape (time, channels), and Asteroid expects (batch, channels, time)
mixture = mixture.transpose()
mixture = mixture.reshape(1, mixture.shape[0], mixture.shape[1])
out_wavs = model.separate(mixture)

# Or simply a file name:
model.separate("Data 1 Target1No 1Cycle No 1Group Depth500m_Data 2 Target2No 10Cycle No 6Group Depth1200m.wav", resample = True)