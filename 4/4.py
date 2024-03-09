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

def check_filter(rate: int, duration: int, volume: float, 
                 ftype: str, forder: int, freqs, detailed_data: bool = False):
    N = duration*rate
    x = np.random.rand(N, 2)
    b, a = signal.butter(forder, freqs, btype=ftype, fs=rate)
    y = signal.lfilter(b ,a, x, axis=0)
    output_signal = amplitude_norm(y, volume)
    if detailed_data:
        filter(b, a, sample_rate)
        afr(output_signal[:,0], sample_rate)
    return output_signal

def ambulance_siren(rate: int, duration: int, volume: float, eps: float):
    N = rate*duration
    f1, f2 = 614, 1844
    span = 0.5
    dn, up = 1 - eps, 1 + eps

    signal_low = check_filter(sample_rate, duration, volume, "bandpass", 2, [dn*f1, up*f1])
    signal_high = check_filter(sample_rate, duration, volume, "bandpass", 2, [dn*f2, up*f2])

    result = np.zeros_like(signal_low, dtype=np.float64)
    up = True
    for i in range(1, N+1):
        if up:
            result[i:,] = signal_high[i:,]
        else:
            result[i:,] = signal_low[i:,]
        if i % (sample_rate*span) == 0:
            up = not up
            print(f"Прогресс: {i/N * 100:.2f}%")
        result[i:,] *= np.exp(-2*i/N)
    result = amplitude_norm(result, 0.5)
    return result

#------------------ Генерация звука (фиолетовый шум) -------------------
sample_rate, duration, vol = 44100, 10, 0.5
output_signal = check_filter(sample_rate, duration, vol, "high", 2, 2200, detailed_data=True)
visualize(output_signal, sample_rate)
io.wavfile.write("4/violet_noise.wav", sample_rate, output_signal)
plt.show()

#------------------ Генерация звука (право <=> лево) -------------------
sample_rate, duration = 44100, 10
output_signal = left_right_ocean(sample_rate, duration, detailed_data=True)
visualize(output_signal, sample_rate)
io.wavfile.write("4/ocean_noise.wav", sample_rate, output_signal)
plt.show()

#------------------ Генерация звука (сирена скорой помощи) -------------------
sample_rate, input_signal = io.wavfile.read("4/siren_sample.wav")
afr(input_signal[:, 0], sample_rate)
visualize(input_signal, sample_rate)
plt.show()
sample_rate, duration = 44100, 10
output_signal = ambulance_siren(sample_rate, duration, 0.5, 0.01)
afr(output_signal[:,0], sample_rate)
visualize(output_signal, sample_rate)
io.wavfile.write("4/ambulance_noise.wav", sample_rate, output_signal.astype(np.int16))
plt.show()