import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

# Функция генерации звука:
def generate_signal(duration: int, freq: int):
    # Задаем параметры для генерации звука
    sample_rate = 44100     # частота дискретизации
    td = 1/sample_rate      # период дискретизации

    # Генерируем звуковой сигнал для левого уха (эталонный)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal_left = np.sin(2 * np.pi * freq * t)

    # Меняем амплитуду
    for i in range(int(sample_rate * duration)):
        signal_left[i] = signal_left[i] * i/int(sample_rate * duration)

    # Генерируем звуковой сигнал для правого канала (копируем эталонный)
    signal_right = signal_left[::-1]

    # Создаем стерео звуковой сигнал
    stereo_signal = np.column_stack((signal_left, signal_right))

    return stereo_signal, sample_rate

# Функция вычисления и визуализации АЧХ
def frec_magni_graph(signal, freq: int):
    # Вычисление амплитудного спектра сигнала
    fft_spectrum = np.fft.fft(signal)
    magnitude_spectrum = 20 * np.log(np.abs(fft_spectrum))

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


#------------- Генерация звука в файл ------------

# Генерация звука
signal, rate = generate_signal(5, 800)

# Воспроизводим стерео звук
sd.play(signal, rate)
sd.wait()

# Сохраняем в файл
sf.write("output_audio.wav", signal, rate)

#--------- Открываем файл для визуализации ------------
signal, sample_rate = sf.read("output_audio.wav")
t = [i/sample_rate for i in range(len(signal))]
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label=['Левое ухо', "Правое ухо"], alpha=0.6)
plt.legend()
plt.ylabel("Максимальная амплитуда")
plt.xlabel("Время (с)")
plt.title("График сигнала для разных каналов")

#---------- Вычисление и визуализация АЧХ ------------
signal, sample_rate = sf.read("output_audio.wav")

frec_magni_graph(signal[:,0], sample_rate)

plt.show()
