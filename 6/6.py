import cv2
import numpy as np
import matplotlib.pyplot as plt

# Функция вычисления спектра изображения
def calcspec(img, for_graph: bool = False):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    S = np.fft.fftshift(np.fft.fft2(img))
    A = np.abs(S)
    A_max = np.max(A)
    eps = A_max * 10**(-6)
    A_dB = 20 * np.log10(A + eps)
    if for_graph:
        return A_dB
    return S, eps

# Функция построения изображения + спектра
def image_spectre(img, spectre_calc) -> None:
    plt.figure(figsize=(14, 7))
    plt.figtext(0.5, 0.95, 'Картинка и её спектр', ha='center', va='center', fontsize=14)

    plt.subplot(121)
    plt.title('Исходное изображение')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(122)
    if isinstance(spectre_calc, bool):
        plt.imshow(calcspec(img, True), cmap='jet')
    else:
        _, eps = calcspec(img)
        plt.imshow(20*np.log10(np.abs(spectre_calc) + eps), cmap='jet')
    plt.title('Спектр изображения')
    plt.colorbar()

# Функция сравнения 2 картинок
def compare2imgs(img1, img2) -> None:
    plt.figure(figsize=(14, 7))
    plt.figtext(0.5, 0.95, 'Сравнение результатов фильтра', ha='center', va='center', fontsize=14)

    plt.subplot(121)
    plt.title('Исходное изображение')
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(122)
    plt.title('После фильтра')
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB))
    plt.axis('off')

# Функция реализации Фурье-фильтрации изображения
def rectfilter(img, param, plot: bool = False):
    
    # Вычисление спектра изображения + размеры
    spectre_input, _ = calcspec(img)
    M, N = spectre_input.shape

    # Пространственная частота + маска фильтрации
    space_freq = param
    mask = np.ones_like(spectre_input)

    # Создание прямоугольной области
    mask[round(M/2) - space_freq: round(M/2) + space_freq,
         round(N/2) - space_freq: round(N/2) + space_freq] = 0
    
    # Применение маски
    spectre_output = spectre_input * mask

    # Обратное преобразование
    J = np.real(np.fft.ifft2(np.fft.ifftshift(spectre_output)))

    # Нормировка значений пикселей
    max_J = np.max(J)
    output = np.uint8((J / max_J) * 255)
    # output = 255 - output

    if plot:
        image_spectre(output, spectre_output)
    return output


#------------------ Основная функция -------------------
I = cv2.imread('6/porsche.jpg', cv2.IMREAD_COLOR)
image_spectre(I, False)

# Фильтр выделение краёв объекта
I2 = rectfilter(I, 40, plot=False)
compare2imgs(I,I2)
plt.show()
'''
TODO:
    [] Переписать из методы программы
    [] Реализовать двумерные ФНЧ, ПФ, РФ (один из трех)
'''