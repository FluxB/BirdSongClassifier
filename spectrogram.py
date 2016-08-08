import librosa
import numpy as np
import sys
import matplotlib.pyplot as plt


def spectrogram(y):
    spec = librosa.core.stft(y)
    return np.abs(spec)


def spectrogram_cqt(y, sr):
    spec = librosa.core.cqt(y, sr=sr)
    return spec


# python3 spectrogram.py spectrogram_files.txt
# where spectrogram_files.txt contains the path to 
# the audio samples that should be visualized
if __name__ == "__main__":
    f_audio = open(str(sys.argv[1]), "r")
    for line in f_audio:
        line = line.strip()
        y, sr = librosa.load(line, offset=0.0, duration=10.0)
        spec = spectrogram(y)
        librosa.display.specshow(librosa.logamplitude(spec**2,
                                                     ref_power=np.max),
                                                     y_axis='linear', x_axis='time')
        plt.title(line)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
