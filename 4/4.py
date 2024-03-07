import numpy as np
from scipy import signal, io
import matplotlib.pyplot as plt

def visualize(signal, sample_rate):
    t = [i/sample_rate for i in range(len(signal))]
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label=['Левое ухо', "Правое ухо"], alpha=0.6)
    plt.legend()
    plt.ylabel("Максимальная амплитуда")
    plt.xlabel("Время (с)")
    plt.title("График сигнала для разных каналов")

def afr(signal, sample_rate):
    # Вычисление амплитудного спектра сигнала
    fft_spectrum = np.fft.fft(signal)
    magnitude_spectrum = 20 * np.log10(np.abs(fft_spectrum))

    # Генерация частотной оси
    freqs = np.fft.fftfreq(len(fft_spectrum), 1/sample_rate)

    # Построение амплитудного спектра
    plt.figure(figsize=(12, 6))
    plt.plot(freqs[:len(freqs)//2], magnitude_spectrum[:len(freqs)//2])
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Уровень (Дб)')
    plt.title('Амплитудный спектр аудиосигнала')
    plt.grid()
    plt.semilogx()

def filter(b, a, rate):
    w, h = signal.freqz(b, a)
    freq = w * rate / (2 * np.pi)
    plt.semilogx(freq, 20 * np.log10(abs(h)))
    plt.title(f"АЧХ фильтра")
    plt.ylabel('Амплитуда (Дб)')
    plt.xlabel('Частоты (Гц)')
    plt.grid()

def amplitude_norm(signal, volume: float):
    """Функция нормировки сигнала по амплитуде

    Parameters
    ----------
    signal : Двумерный массив
        Исходный сигнал, которому необходимо отрегулировать громкость
    volume : float (0, 1)
        Требуемая громкость в процентах

    Returns
    -------
    output_signal: Двумерный массив
        Сигнал с изменённой громкостью
    """
    maxY, minY = np.max(signal, axis=0), np.min(signal, axis=0)
    norm1, norm2 = abs(minY[0]), abs(minY[0])
    if maxY[0] > np.abs(minY[0]): norm1 = maxY[0]
    if maxY[1] > np.abs(minY[1]): norm2 = maxY[1]

    y = np.zeros_like(signal, dtype=np.float16)
    y[:, 0] = 32767*volume * (signal[:, 0] / norm1)
    y[:, 1] = 32767*volume * (signal[:, 1] / norm2)

    output_signal = y.astype(np.int16)
    return output_signal

def left_right_ocean(rate, duration, detailed_data: bool = False):
    N = duration*rate
    x1 = np.random.rand(N, 2)
    f1_low, f1_up = 400, 900
    b, a = signal.butter(2, [f1_low, f1_up], btype='bandpass', fs=sample_rate)
    y1 = signal.lfilter(b, a, x1, axis=0)
    y2 = np.zeros_like(y1)
    for i in range(N):
        y2[i,0] = np.sin(2*np.pi*i/N)
        y2[i,1] = np.cos(2*np.pi*i/N)
    y12 = y1 * y2
    output_signal = amplitude_norm(y12, 0.5)

    if detailed_data:
        filter(b, a, sample_rate)
        afr(output_signal[:,0], sample_rate)

    return output_signal


#------------------ Генерация звука (право <=> лево) -------------------
sample_rate, duration = 44100, 10
output_signal = left_right_ocean(sample_rate, duration, detailed_data=False)
visualize(output_signal, sample_rate)

plt.show()
io.wavfile.write("4/test.wav", sample_rate, output_signal)

#------------------ Генерация звука () -------------------
N = sample_rate*duration

x = np.random.rand(N, 2) * 2 - 0.5
eps1 = 0.01
dn1, up1 = 1 - eps1, 1 + eps1
f1 = 880
b, a = signal.butter(2, [dn1*f1, up1*f1], btype='bandpass', fs=sample_rate)
y1 = signal.lfilter(b, a, x, axis=0)

eps2 = 0.1
dn2, up2 = 1 - eps2, 1 + eps2
f2 = 5*f1
b, a = signal.butter(2, [dn2*f2, up2*f2], btype='bandpass', fs=sample_rate)
y2 = signal.lfilter(b, a, x, axis=0)

y12 = np.zeros_like(y1)

for i in range(N):
    alfa = 1/(1 + np.exp(-10*(2*(i-N/2)/N)))
    y12[i:,] = y1[i:,] * (1-alfa) + y2[i:,] * alfa
    y12[i:,] = y12[i:,] * np.exp(-i/N)*np.cos(6*np.pi*i/N)

output_signal = amplitude_norm(y12, 0.5)
visualize(output_signal, sample_rate)
io.wavfile.write("4/test2.wav", sample_rate, output_signal)
plt.show()
'''
TODO:
[] Изучить шаблоны в методе, изменить параметры
[] Создать собственный шумоподобный сигнал (метод формирующего фильтра)
[] Дополнительно: сформировать реалистичный звук таким же методом
'''