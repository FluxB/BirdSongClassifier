import librosa
import numpy as np
import sys
import matplotlib.pyplot as plt


def spectrogram(y):
    spec = librosa.core.stft(y, n_fft=512)
    return np.abs(spec)


def spectrogram_cqt(y, sr):
    spec = librosa.core.cqt(y, sr=sr)
    return spec

def plot_spect(spec):
    plt.figure(figsize=(12, 8))
    nb=len(spec)
    i=0
    for s in spec:
        i+=1
        plt.subplot(nb, 1, i)
        D = librosa.logamplitude(np.abs(s)**2, ref_power=np.max)
        librosa.display.specshow(D,y_axis='log', x_axis='time')
    plt.show()

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
        plt.imshow(librosa.logamplitude(spec**2,
                                        ref_power=np.max),
                                        interpolation='none')
        plt.show()
