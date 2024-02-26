import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

# Функция создания АЧХ
def amplitude_frequency_response(signal, sample_rate):
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

# Функция фильтра высоких частот
def high_pass_filter(min_freq: int, signal, sample_rate):
    # Переводим Гц в индексы массива
    n_min_freq = int(len(signal) * min_freq/sample_rate)

    # Создаем спектр входного и выходного сигнала
    fft_spectrum_input = np.fft.fft2(signal)
    fft_spectrum_output = np.zeros_like(fft_spectrum_input) + 10**(-10)

    # Реализация ФВЧ
    for i in range(1, n_min_freq):
        fft_spectrum_output[i, :] = fft_spectrum_input[i, :]
        fft_spectrum_output[len(signal)-i, :] = fft_spectrum_input[len(signal)-i, :]
    fft_spectrum_output[0, :] = fft_spectrum_input[0, :]
    fft_spectrum_output = fft_spectrum_input - fft_spectrum_output

    # Обратное преобразование в звук
    output_signal = np.real(np.fft.ifft2(fft_spectrum_output))

    return output_signal

# -------------------------------------------------------------------------
signal, sample_rate = sf.read("2/input_audio.wav")
amplitude_frequency_response(signal[:,0], sample_rate)

signal = high_pass_filter(500, signal, sample_rate)
amplitude_frequency_response(signal[:,0], sample_rate)

plt.show()
sf.write("2/output_audio.wav", signal, sample_rate)


'''
TODO:
[X] Создать функцию формирования АЧХ (как в первой лабе)
[] Написать реализацию ФВЧ(800 Гц) в виде функции 
[] Показать работу функции на примере радио из портал
[] Создать запись голоса и добавить на неё звенящую помеху
[] Написать реализацию режекторного фильтра в виде функции для удаления помехи
'''