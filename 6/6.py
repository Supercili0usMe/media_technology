import cv2
import numpy as np
import matplotlib.pyplot as plt

# Функция вычисления спектра изображения
def calcspec(img, for_graph: bool = False):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(float)
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
        A = np.abs(spectre_calc)
        A_max = np.max(A)
        eps = A_max * 10 ** (-6)
        plt.imshow(20*np.log10(np.abs(spectre_calc) + eps), cmap='jet')
    plt.title('Спектр изображения')
    plt.colorbar()

# Функция сравнения 2 картинок
def compare2imgs(img1, img2, title1: str, title2: str, main_title: str) -> None:
    plt.figure(figsize=(14, 7))
    plt.figtext(0.5, 0.95, main_title, ha='center', va='center', fontsize=14)

    plt.subplot(121)
    plt.title(title1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(122)
    plt.title(title2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB))
    plt.axis('off')

# Функция реализации Фурье-фильтрации изображения (квадрат)
def squarefilter(img, param, plot: bool = False):
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
    J = np.abs(np.fft.ifft2(np.fft.ifftshift(spectre_output)))

    # Нормировка значений пикселей
    max_J = np.max(J)
    output = np.uint8((J / max_J) * 255)
    output = 255 - output

    if plot:
        image_spectre(output, spectre_output)
    return output

# Функция реализации Фурье-фильтрации изображения (колокол)
def bellfilter(img, param, plot: bool = False):
    # Вычисление спектра изображения + размеры
    spectre_input, _ = calcspec(img)
    x, y = spectre_input.shape

    # Пространственная частота + маска фильтрации
    space_freq = param
    mask = np.ones_like(spectre_input)

    # Создание маски
    cx, cy = round(x/2), round(y/2)
    nx, ny = np.meshgrid(np.arange(x), np.arange(y))
    mask = 1 - (np.exp(-((cx - nx) / space_freq) ** 2) * np.exp(-((cy - ny) / space_freq) ** 2))
    
    # Применение маски
    spectre_output = spectre_input * mask

    # Обратное преобразование
    J = np.abs(np.fft.ifft2(np.fft.ifftshift(spectre_output)))

    # Нормировка значений пикселей
    max_J = np.max(J)
    output = np.uint8((J / max_J) * 255)
    output = 255 - output

    if plot:
        image_spectre(output, spectre_output)
    return output

# Функция пространственного ФВЧ Баттерворта
def newfilter(img, D_up, K, plot: bool = False):
    # Вычисление спектра изображения
    S_I, _ = calcspec(img)

    # Начальные параметры
    M, N = S_I.shape
    Cn = round(N / 2)
    Cm = round(M / 2)

    # Формирование массива d
    nx, ny = np.meshgrid(np.arange(N), np.arange(M))
    d = np.sqrt((nx - Cn) ** 2 + (ny - Cm) ** 2)

    # Формирование массива W
    deg = 2 * K
    W = (d/ D_up) ** K / np.sqrt(1 + (d/ D_up) ** deg)
    
    # Применение фильтра к спектру
    S_J = S_I * W

    # Обратное преобразование Фурье
    J = np.abs(np.fft.ifft2(np.fft.ifftshift(S_J)))

    # Нормализация и инверсия изображения
    max_J = np.max(J)
    output_image = (255 * (J / max_J)).astype(np.uint8)
    output_image = 255 - output_image

    if plot:
        image_spectre(output_image, S_J)
    return output_image

# Функция запуска предустановленных фильтров
def presetfilters(img, ftype: str, ddepth = cv2.CV_16S, ksize=3, threshold1=100, threshold2 = 200):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    match ftype:
        case "Sobel":
            grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=ksize)
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        case "Canny":
            edges = cv2.Canny(img, threshold1=threshold1, threshold2=threshold2).astype(np.uint8)
    edges = 255 - edges
    return edges

#------------------ Основная функция -------------------
I = cv2.imread('6/porsche.jpg', cv2.IMREAD_COLOR)
image_spectre(I, False)

# Фильтр выделения краёв объекта (область-квадрат)
I2 = squarefilter(I, 30, False)
compare2imgs(I, I2, "Исходное изображение", "После применения фильтра", "Результаты работы фильтра")

# Фильтр выделения краёв объекта (область-колокол)
I21 = bellfilter(I, 30, False)
compare2imgs(I2, I21, "Квадратная область", "Колоколообразная область", "Сравнение двух разных областей")

# Фильтр выделения краев объекта (ФВЧ Баттерворта)
I3 = newfilter(I, 50, 4, False)
compare2imgs(I21, I3, "Колоколообразная область", "пространственный ФВЧ", "Сравнение двух разных фильтров")
compare2imgs(I2, I3, "Квадратная область", "пространственный ФВЧ", "Сравнение двух разных фильтров")

# Реализованные фильтры (Sobel, Canny)
I4 = presetfilters(I, "Sobel")
compare2imgs(I2, I4, "Квадратная область", "Фильтр Собеля", "Сравнение двух разных фильтров")

I5 = presetfilters(I, "Canny")
compare2imgs(I2, I5, "Квадратная область", "Фильтр Кенни", "Сравнение двух разных фильтров")

# Смотрим на работу лучшего фильтра на сложной картинке
H = cv2.imread('6/st_petersburg.jpg', cv2.IMREAD_COLOR)
H2 = presetfilters(H, "Sobel")
compare2imgs(H, H2, "Исходное изображение", "Фильтр Собеля", "Результаты работы фильтра")


plt.show()
