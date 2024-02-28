import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Функция создания АЧХ
def amplitude_frequency_response(signal, sample_rate, title: str):
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
    plt.title(f'Амплитудный спектр аудиосигнала {title}')
    plt.grid()
    plt.semilogx()

# Функция ФВЧ
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

# Функция РФ
def notch_filter(min_freq: int, max_freq: int, signal, sample_rate):
    # Переводим Гц в индексы массива
    n_min_freq = int(len(signal) * min_freq/sample_rate)
    n_max_freq = int(len(signal) * max_freq/sample_rate)

    # Создаем спектр входного и выходного сигнала
    fft_spectrum_input = np.fft.fft2(signal)
    fft_spectrum_output_LOW = np.zeros_like(fft_spectrum_input) + 10**(-10)
    fft_spectrum_output_HIGH = np.zeros_like(fft_spectrum_input) + 10**(-10)

    # Реализация РФ
    for i in range(1, n_min_freq):
        fft_spectrum_output_LOW[i, :] = fft_spectrum_input[i, :]
        fft_spectrum_output_LOW[len(signal)-i, :] = fft_spectrum_input[len(signal)-i, :]
    for i in range(1, n_max_freq):
        fft_spectrum_output_HIGH[i, :] = fft_spectrum_input[i, :]
        fft_spectrum_output_HIGH[len(signal)-i, :] = fft_spectrum_input[len(signal)-i, :]
    fft_spectrum_output_LOW[0, :] = fft_spectrum_input[0, :]
    fft_spectrum_output_HIGH[0, :] = fft_spectrum_input[0, :]
    fft_spectrum_output_HIGH -= fft_spectrum_input
    fft_spectrum_output = fft_spectrum_output_LOW + fft_spectrum_output_HIGH

    # Обратное преобразование в звук
    output_signal = np.real(np.fft.ifft2(fft_spectrum_output))

    return output_signal

# Функция добавления помехи
def create_noise(freq: int, amplitude: float, signal, sample_rate: int):
    duration = len(signal) / sample_rate

    # Генерируем звуковой сигнал
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    noise = np.sin(2 * np.pi * freq * t) * amplitude
    final_noise = np.column_stack((noise, noise))
    noisy_signal = signal + final_noise

    return noisy_signal


# ------------------- Работа с ФНЧ -----------------------------
# Считываем файл и создаем АЧХ
signal, sample_rate = sf.read("2/input_audio.wav")
amplitude_frequency_response(signal[:,0], sample_rate, '(исходный)')

# Пропускаем сигнал через ФВЧ и создаём АЧХ
signal = high_pass_filter(800, signal, sample_rate)
amplitude_frequency_response(signal[:,0], sample_rate, '(после ФВЧ)')

# Выводим получившиеся рисунки + сохраняем в файл
plt.show()
sf.write("2/output_audio.wav", signal, sample_rate)

# ------------------- Работа с РФ -----------------------------
# Считываем файл и создаем АЧХ
signal, sample_rate = sf.read("2/extra_task_input.wav")
amplitude_frequency_response(signal[:,0], sample_rate, '(исходный)')

# Добавляем помеху, создаём АЧх, сохраняем файл
signal = create_noise(12000, 0.05, signal, sample_rate)
amplitude_frequency_response(signal[:,0], sample_rate, '(c помехой)')
sf.write("2/extra_task_noisy.wav", signal, sample_rate)

# Пропускаем сигнал через РФ и создаём АЧХ
signal = notch_filter(9000, 17000, signal, sample_rate)
amplitude_frequency_response(signal[:,0], sample_rate, '(после РФ)')

# Выводим получившиеся рисунки + сохраняем в файл
plt.show()
sf.write("2/extra_task_cleaned.wav", signal, sample_rate)
