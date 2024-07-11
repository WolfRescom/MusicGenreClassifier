import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

file = "Data/genres_original/blues/blues_0000.wav"

signal, sr = librosa.load(file, sr = 22050)
# librosa.display.waveshow(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0,sr, len(magnitude))
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]
# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

n_fft = 2048
hop_length = 512
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
