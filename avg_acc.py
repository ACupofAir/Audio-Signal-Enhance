import os
import torch
import torchaudio
from asteroid.losses import PITLossWrapper
from asteroid.losses.sdr import PairwiseNegSDR



def get_snri(device, checkpoint, audio_input_path, output_folder_path):

    model = torch.load(checkpoint).to(device)
    model.eval()
    with torch.no_grad():
        model.separate(
            audio_input_path,
            output_dir=output_folder_path,
            resample=True,
        )

    audio_output_path = os.path.join(
        output_folder_path,
        os.path.basename(audio_input_path).replace(".wav", "_est1.wav"),
    )

    audio_input, _ = torchaudio.load(audio_input_path)
    audio_input = audio_input.unsqueeze(0)
    audio_output, _ = torchaudio.load(audio_output_path)
    audio_output = audio_output.unsqueeze(0)

    # Ensure the audio tensors are on the same device as the model
    audio_input = audio_input.to(device)
    audio_output = audio_output.to(device)
    loss_func = PITLossWrapper(PairwiseNegSDR("sisdr"), pit_from="pw_mtx").to(
        device
    )
    loss = loss_func(audio_output, audio_input)
    return -1 * loss.item()
    
if __name__== "__main__":
    acc_arr = []
    ckp = r"C:\Users\ASUS\junwang\Asteroid\checkpoint\2024-12-02\shipsear_5.pth"
    # input_folder = r"C:\Users\ASUS\junwang\Asteroid\data_backup\data_shipsear_5\val\mix_both"
    input_folder = r"C:\Users\ASUS\junwang\Asteroid\data\val\mix_both"
    output_folder = r"C:\Users\ASUS\junwang\Asteroid\code\output"
    input_files = os.listdir(input_folder)
    for input_file in input_files:
        input_file = os.path.join(input_folder, input_file)
        # print(input_file)
        acc_arr.append(get_snri('cuda',ckp,input_file, output_folder))
    print(len(acc_arr))
    acc_lt_zero = [x for x in acc_arr if x <3]
    print(len(acc_lt_zero))
    print(acc_lt_zero)
    min_acc = min(acc_arr)
    max_acc = max(acc_arr)
    avg_acc = sum(acc_arr)/len(acc_arr)
    print("min_acc:", min_acc)
    print("max_acc:", max_acc)
    print("avg_acc:", avg_acc)