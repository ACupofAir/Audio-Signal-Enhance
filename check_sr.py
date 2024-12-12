import wave
audio_file = r"C:\Users\ASUS\junwang\Asteroid\data\train\s1\3.wav"
with wave.open(audio_file, 'rb') as wav_file:
    sample_rate = wav_file.getframerate()
    print(sample_rate)
