# python check_sr.py <audio_file>
import wave
import sys

audio_file = sys.argv[1]

with wave.open(audio_file, "rb") as wav_file:
    sample_rate = wav_file.getframerate()
    print("Input file: ", audio_file)
    print("Sample rate: ", sample_rate)
