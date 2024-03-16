import cv2 as cv
import numpy as np
from skimage import exposure, color
import matplotlib.pyplot as plt


def plot_image_brightness(img, title: str) -> None:
    """Функция для построения на одной фигуре исходного
    изображения и её гистограммы яркости

    Parameters
    ----------
    img : matrix
        Матрица цветов изображения
    title : str
        Подпись к общей фигуре
    """
    plt.figure(figsize=(14, 7))
    plt.figtext(0.5, 0.95, title, ha='center', va='center', fontsize=14)
    plt.subplot(121)
    plt.title("Изображения")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(122)
    plt.title('Гистограмма яркости изображения')
    plt.hist(cv.cvtColor(img, cv.COLOR_BGR2GRAY).ravel(), bins=256, range=[0, 256], edgecolor="black", linewidth=0.2)

def compare2image(img1, img2, title: str) -> None:
    plt.figure(figsize=(14, 7))
    plt.figtext(0.5, 0.95, title, ha='center', va='center', fontsize=14)

    plt.subplot(221)
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(222)
    plt.hist(cv.cvtColor(img1, cv.COLOR_BGR2GRAY).ravel(), bins=256, range=[0, 256], edgecolor="black", linewidth=0.2)
    plt.xlabel("До применения")

    plt.subplot(223)
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(224)
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY).ravel()
    if np.max(gray) <= 1.0:
        plt.hist(cv.cvtColor(img2 * 256, cv.COLOR_BGR2GRAY).ravel(), bins=256, range=[0, 256], edgecolor="black", linewidth=0.2)
    else:
        plt.hist(cv.cvtColor(img2, cv.COLOR_BGR2GRAY).ravel(), bins=256, range=[0, 256], edgecolor="black", linewidth=0.2)
    plt.xlabel("После применения")

def compare6image(imgs, params) -> None:
    fig, axs = plt.subplots(2, 3, num="Изображения", figsize=(14, 7))
    fig.suptitle("Сравнение качества изображения")
    for i, ax in enumerate(axs.flat):
        ax.imshow(cv.cvtColor(imgs[i], cv.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f"Ядро: {params[i]}")
    fig.tight_layout()
    
    fig, axs = plt.subplots(2, 3, num="гистограммы яркости", figsize=(14, 7))
    fig.suptitle("Сравнение гистограмм яркости")
    for i, ax in enumerate(axs.flat):
        ax.hist(cv.cvtColor(imgs[i] * 256, cv.COLOR_BGR2GRAY).ravel(), bins=256, range=[0, 256], edgecolor="black", linewidth=0.2)
        ax.set_title(f"Ядро: {params[i]}")
    fig.tight_layout()

def image_histeq(img):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    j_hsv = np.copy(img_hsv)
    V = img_hsv[:,:,2]          # Выделение матрицы яркости
    V_out = cv.equalizeHist(V)  # Эквализация гистограммы
    j_hsv[:,:,2] = V_out
    j = cv.cvtColor(j_hsv, cv.COLOR_HSV2BGR)
    return j

def image_imadjust(image, gamma=1.0, low_percentile=1, high_percentile=99):
    # Растяжение диапазона интенсивностей
    p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
    img_rescale = exposure.rescale_intensity(image, in_range=(p_low, p_high))

    # Гамма-коррекция
    img_gamma = exposure.adjust_gamma(img_rescale, gamma)
    return img_gamma

def image_adapthisteq(img, kernel_size=[8, 8], clip_limit=0.02, convert2lab: bool=False):
    if convert2lab:
        LAB = color.rgb2lab(img)
        L = LAB[:,:,0] / 100
        L = exposure.equalize_adapthist(L, clip_limit=clip_limit, kernel_size=kernel_size)
        LAB[:,:,0] = L * 100
        J = color.lab2rgb(LAB)
    else:
        J = exposure.equalize_adapthist(img, clip_limit=clip_limit, kernel_size=kernel_size)
    return np.float32(J)

#------------------ Обзор функций -------------------
# Первичный осмотр изображений
island = cv.imread('5/mj_kand.png')
plot_image_brightness(island, 'Исходное изображение')

car = cv.imread('5/porsche.jpg')
plot_image_brightness(car, 'Исходное изображение')

landscape = cv.imread('5/1.jpg')
plot_image_brightness(landscape, 'Исходное изображение')
plt.show()

# Применение функции эквализации гистограммы
temp = image_histeq(car)
compare2image(car, temp, "Эквализация (выравнивание) гистограммы яркости")
plt.show()

# Применение функции увеличения контраста
temp = image_imadjust(landscape)
compare2image(landscape, temp, "Увеличение контрастности растяжением диапазона")
plt.show()

# Применение функции контрастно-ограниченного адаптивного выравнивания гистограммы (CLAHE)
temp = image_adapthisteq(island, convert2lab=False)
compare2image(island, temp, "Контрастно-ограниченного адаптивного выравнивания гистограммы")
plt.show()

#------------------ Влияние параметра -------------------
kernels = [(4, i) for i in np.logspace(start=1, stop=6, base=2, num=6, dtype=np.int16)]
results = []
for kernel in kernels:
    print(f'Расчет при размере ядра: {kernel}')
    results.append(image_adapthisteq(landscape, kernel_size=kernel))

compare6image(results, kernels)
plt.show()