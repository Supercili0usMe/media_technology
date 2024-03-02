import numpy as np
from scipy import signal, io
import soundfile as sf
import matplotlib.pyplot as plt

# Функция сравнения двух АЧХ
def afr2(signal_1, signal_2, sample_rate, ftype, order):
    # Вычисление амплитудного спектра сигнала
    fft_spectrum_1 = np.fft.fft(signal_1)
    fft_spectrum_2 = np.fft.fft(signal_2)
    magnitude_spectrum_1 = 20 * np.log10(abs(fft_spectrum_1))
    magnitude_spectrum_2 = 20 * np.log10(abs(fft_spectrum_2))
    # Генерация частотной оси
    freqs = np.fft.fftfreq(len(fft_spectrum_1), 1/sample_rate)
    # Построение амплитудного спектра
    plt.figure(figsize=(12, 6))
    plt.semilogx(freqs[:len(freqs)//2], magnitude_spectrum_1[:len(freqs)//2], label="До фильтра", alpha=0.7)
    plt.semilogx(freqs[:len(freqs)//2], magnitude_spectrum_2[:len(freqs)//2], label="После фильтра", alpha=0.7)
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Уровень (Дб)')
    plt.title(f'Сравнение амплитудных спектров ("{ftype}", порядок = {order})')
    plt.legend()
    plt.grid()

# Функция построения графика фильтра
def filter(b, a, params):
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)), label=f'{params}')
    plt.title(f"АЧХ фильтра")
    plt.ylabel('Амплитуда (Дб)')
    plt.xlabel('Частоты (Гц)')
    plt.grid()
    plt.legend()
    plt.ylim((-105, 5))

# Функция создания РЦФ ФВЧ
def highpass_IIR_filter(low_freq: int, IIR_order: int, filter_type: str, input_signal, sample_rate, plot_ones: bool = False,
                        plot_multiple: bool = False):
    # Расчет коэффициентов в зависимости от типа фильтра:
    rp, rs = 2, 60
    params = {'order': IIR_order, 'ftype': filter_type}
    match filter_type:
        case "butter": 
            sos = signal.butter(IIR_order, low_freq, btype='high', fs=sample_rate, output='sos')
            b, a = signal.butter(IIR_order, low_freq, btype='high', analog=True)
        case "cheby1": 
            sos = signal.cheby1(IIR_order, rp, low_freq, btype='high', fs=sample_rate, output='sos')
            b, a = signal.cheby1(IIR_order, rp, low_freq, btype='high', analog=True)
            params["rp"] = rp
        case "cheby2": 
            sos = signal.cheby2(IIR_order, rs, low_freq, btype='high', fs=sample_rate, output='sos')
            b, a = signal.cheby2(IIR_order, rs, low_freq, btype='high', analog=True)
            params["rs"] = rs
        case "ellip": 
            sos = signal.ellip(IIR_order, rp, rs, low_freq, btype='high', fs=sample_rate, output='sos')
            b, a = signal.ellip(IIR_order, rp, rs, low_freq, btype='high', analog=True)
            params["rp"] = rp
            params["rs"] = rs

    # Применяем фильтрацию к сигналу и строим график
    output_signal = signal.sosfilt(sos, input_signal, axis=0)
    if plot_ones:
        plt.figure(figsize=(12, 6))
        filter(b, a, params)
    if plot_multiple:
        filter(b, a, params)
    return output_signal

# ------------------- Работа с ФВЧ -----------------------------
# Считываем файл и создаем базовые параметры
input_signal, sample_rate = sf.read("3/input_audio.wav")
# sample_rate, input_signal = io.wavfile.read("3/input_audio.wav")
fft_input_spectrum = np.fft.fft(input_signal[:,0])
magnitude_input_spectrum = 20 * np.log10(abs(fft_input_spectrum))
freqs = np.fft.fftfreq(len(fft_input_spectrum), 1/sample_rate)

# Пропускаем сигнал через ФВЧ и сравниваем АЧХ
low_freq, IIR_order, filter_type = 800, 4, 'butter'
filtered_signal= highpass_IIR_filter(low_freq, IIR_order, filter_type, input_signal, sample_rate, plot_ones=True)
afr2(input_signal[:,0], filtered_signal[:,0], sample_rate, filter_type, IIR_order)
# io.wavfile.write("3/output_audio.wav", sample_rate, filtered_signal.astype(np.int16))

plt.show()

# Сравнение всех алгоритмов между собой
filter_type = ['butter', 'cheby1', 'cheby2', 'ellip']
filtered_signal = []
plt.figure(figsize=(15, 8))
for i ,ftype in enumerate(filter_type):
    plt.subplot(2, 2, i+1)
    temp = highpass_IIR_filter(low_freq, IIR_order, ftype, input_signal, sample_rate, plot_multiple=True)
    filtered_signal.append(temp[:, 0])
    # io.wavfile.write(f"3/output_audio_{ftype}.wav", sample_rate, temp.astype(np.int16))
plt.figure(figsize=(15, 8))
for i, s in enumerate(filtered_signal):
    fft_filtered_spectrum = np.fft.fft(s)
    magnitude_filtered_spectrum = 20 * np.log10(abs(fft_filtered_spectrum))
    plt.subplot(2, 2, i+1)
    plt.semilogx(freqs[:len(freqs)//2], magnitude_input_spectrum[:len(freqs)//2], label="До фильтра", alpha=0.7)
    plt.semilogx(freqs[:len(freqs)//2], magnitude_filtered_spectrum[:len(freqs)//2], label="После фильтра", alpha=0.7)
    plt.legend()
    plt.grid()
    plt.title(f'{filter_type[i]}: order = {IIR_order}')
    if i % 2 == 0:
        plt.ylabel("Частота (Гц)")
    if i >= 2:
        plt.xlabel("Уровень (Дб)")

plt.show()

# ------------------- Работа с РФ -----------------------------

'''
Отправить преподу:
https://www.mathworks.com/help/signal/ref/ellip.html

TODO:
[+] Написать рекурсивный цифровой фильтр (РЦФ) в качестве функции
[+] Выполнить ту же последовательность действий что и в №2
[+] Сравнить звучание полученное этим фильтром и предыдущим (ФВЧ)
[+] Дополнительно: сравнить все алгоритмы по звучанию на ФВЧ
'''