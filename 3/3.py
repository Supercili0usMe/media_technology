import numpy as np
from scipy import signal, io
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
def IIR_filter(input_signal, sample_rate, plot_ones: bool = False, plot_multiple: bool = False, **kwargs):
    # Расчет коэффициентов в зависимости от типа фильтра:
    rp, rs = 2, 60
    kwargs = kwargs["kwargs"]
    params = {'order': kwargs['IIR_order'], 'ftype': kwargs['filter_type']}
    match kwargs['filter_type']:
        case "butter": 
            sos = signal.butter(kwargs['IIR_order'], kwargs['low_freq'], btype=kwargs['alg_type'], fs=sample_rate, output='sos')
            b, a = signal.butter(kwargs['IIR_order'], kwargs['low_freq'], btype=kwargs['alg_type'], analog=True)
        case "cheby1": 
            sos = signal.cheby1(kwargs['IIR_order'], rp, kwargs['low_freq'], btype=kwargs['alg_type'], fs=sample_rate, output='sos')
            b, a = signal.cheby1(kwargs['IIR_order'], rp, kwargs['low_freq'], btype=kwargs['alg_type'], analog=True)
            params["rp"] = rp
        case "cheby2": 
            sos = signal.cheby2(kwargs['IIR_order'], rs, kwargs['low_freq'], btype=kwargs['alg_type'], fs=sample_rate, output='sos')
            b, a = signal.cheby2(kwargs['IIR_order'], rs, kwargs['low_freq'], btype=kwargs['alg_type'], analog=True)
            params["rs"] = rs
        case "ellip": 
            sos = signal.ellip(kwargs['IIR_order'], rp, rs, kwargs['low_freq'], btype=kwargs['alg_type'], fs=sample_rate, output='sos')
            b, a = signal.ellip(kwargs['IIR_order'], rp, rs, kwargs['low_freq'], btype=kwargs['alg_type'], analog=True)
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

# Функция сравнения всех алгоритмов
def compare_filters(input_signal, sample_rate, save_audio: bool = False, **kwargs):
    params = kwargs["kwargs"]
    filter_type = ['butter', 'cheby1', 'cheby2', 'ellip']
    filtered_signal = []
    plt.figure(figsize=(15, 8))
    for i ,ftype in enumerate(filter_type):
        params["filter_type"] = ftype
        plt.subplot(2, 2, i+1)
        temp = IIR_filter(input_signal, sample_rate, plot_multiple=True, kwargs=params)
        filtered_signal.append(temp[:, 0])
        if save_audio:
            io.wavfile.write(f"3/output_audio_{ftype}.wav", sample_rate, temp.astype(np.int16))
    plt.figure(figsize=(15, 8))
    for i, s in enumerate(filtered_signal):
        fft_filtered_spectrum = np.fft.fft(s)
        magnitude_filtered_spectrum = 20 * np.log10(abs(fft_filtered_spectrum))
        plt.subplot(2, 2, i+1)
        plt.semilogx(freqs[:len(freqs)//2], magnitude_input_spectrum[:len(freqs)//2], label="До фильтра", alpha=0.7)
        plt.semilogx(freqs[:len(freqs)//2], magnitude_filtered_spectrum[:len(freqs)//2], label="После фильтра", alpha=0.7)
        plt.legend()
        plt.grid()
        plt.title(f'{filter_type[i]}: order = {params["IIR_order"]}')
        if i % 2 == 0:
            plt.ylabel("Частота (Гц)")
        if i >= 2:
            plt.xlabel("Уровень (Дб)")

# ------------------- Работа с ФВЧ -----------------------------
# Считываем файл и создаем базовые параметры
sample_rate, input_signal = io.wavfile.read("3/input_audio.wav")
fft_input_spectrum = np.fft.fft(input_signal[:,0])
magnitude_input_spectrum = 20 * np.log10(abs(fft_input_spectrum))
freqs = np.fft.fftfreq(len(fft_input_spectrum), 1/sample_rate)

# Пропускаем сигнал через ФВЧ и сравниваем АЧХ
params = {}
params["low_freq"] = 800
params["alg_type"] = 'high'
params["IIR_order"] = 4
params["filter_type"] = 'butter'
filtered_signal = IIR_filter(input_signal, sample_rate, plot_ones=True, kwargs=params)
afr2(input_signal[:,0], filtered_signal[:,0], sample_rate, params["filter_type"], params["IIR_order"])
io.wavfile.write("3/output_audio.wav", sample_rate, filtered_signal.astype(np.int16))
plt.show()

# Сравнение всех алгоритмов между собой
compare_filters(input_signal, sample_rate, save_audio=False, kwargs=params)
plt.show()

# ------------------- Работа с РФ -----------------------------
# Считываем файл и создаем его базовые характеристики
sample_rate, input_signal = io.wavfile.read("3/extra_task_noisy.wav")
fft_input_spectrum = np.fft.fft(input_signal[:,0])
magnitude_input_spectrum = 20 * np.log10(abs(fft_input_spectrum))
freqs = np.fft.fftfreq(len(fft_input_spectrum), 1/sample_rate)

# Пропускаем сигнал через РФ и сравниваем АЧХ
params = {}
params["low_freq"] = [9000, 17000]
params["alg_type"] = 'stop'
params["IIR_order"] = 4
params["filter_type"] = 'butter'
filtered_signal= IIR_filter(input_signal, sample_rate, plot_ones=True, kwargs=params)
afr2(input_signal[:,0], filtered_signal[:,0], sample_rate, params["filter_type"], params["IIR_order"])
io.wavfile.write("3/extra_task_cleaned.wav", sample_rate, filtered_signal.astype(np.int16))
plt.show()

# Сравнение всех алгоритмов между собой
compare_filters(input_signal, sample_rate, save_audio=False, kwargs=params)
plt.show()