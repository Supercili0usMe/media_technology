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
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.axis('off')



# Повышение фокусировки фильтром Лапласа
def laplas_filter(img, c: float):
    kernel = np.array([
        [-c, -c, -c],
        [-c, 1 + 8 * c, -c],
        [-c, -c, -c]
    ])
    filtered_image = cv2.filter2D(img, -1, kernel)
    return filtered_image


#------------------ Основная функция -------------------
blured_city = cv2.imread('7/st_petersburg.jpg', cv2.IMREAD_COLOR)

# Применение фильтра Лапласа
c = 0.5
remove_blured_city = laplas_filter(blured_city, c)
compare2imgs(blured_city, remove_blured_city, 'Исходная картинка', 
             f'Фильтр Лапласа ({c = })', 'Сравнение результата повышения четкости')

# plt.show()

def generate_sgrid(p2):
    rad = p2
    crad = int(np.ceil(rad - 0.5))
    
    x_min, x_max = -crad, crad
    y_min, y_max = -crad, crad
    
    x, y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    maxxy = np.maximum(np.abs(x), np.abs(y))
    minxy = np.minimum(np.abs(x), np.abs(x))
    
    m1 = (rad**2 < (maxxy+0.5)**2 + (minxy-0.5)).astype(float) * (minxy-0.5) + \
         (rad**2 >= (maxxy+0.5)**2 + (minxy-0.5)).astype(float) * np.sqrt(rad**2 - (maxxy + 0.5)**2)
    
    m2 = (rad**2 > (maxxy-0.5)**2 + (minxy+0.5)).astype(float) * (minxy+0.5) + \
         (rad**2 <= (maxxy-0.5)**2 + (minxy+0.5)).astype(float) * np.sqrt(rad**2 - (maxxy - 0.5)**2)
    
    term1 = rad**2 * (0.5 * (np.arcsin(m2/rad) - np.arcsin(m1/rad)) + 0.25 * (np.sin(2*np.arcsin(m2/rad)) - np.sin(2*np.arcsin(m1/rad))))
    term2 = (maxxy-0.5) * (m2-m1)
    term3 = (m1-minxy+0.5)
    
    mask = (((rad**2 < (maxxy+0.5)**2 + (minxy+0.5)**2) & (rad**2 > (maxxy-0.5)**2 + (minxy-0.5)**2)) | ((minxy==0) & (maxxy-0.5 < rad) & (maxxy+0.5>=rad)))
    
    sgrid = (term1 - term2 + term3) * mask
    sgrid += ((maxxy+0.5)**2 + (minxy+0.5)**2 < rad**2)
    
    sgrid[crad+1, crad+1] = min(np.pi*rad**2, np.pi/2)
    
    if crad > 0 and rad > crad-0.5 and rad**2 < (crad-0.5)**2+0.25:
        m1 = np.sqrt(rad**2 - (crad - 0.5)**2)
        m1n = m1/rad
        sg0 = 2 * (rad**2 * (0.5*np.arcsin(m1n) + 0.25*np.sin(2*np.arcsin(m1n))) - m1 * (crad-0.5))
        sgrid[2*crad+1, crad+1] = sg0
        sgrid[crad+1, 2*crad+1] = sg0
        sgrid[crad+1, 1] = sg0
        sgrid[1, crad+1] = sg0
        sgrid[2*crad, crad+1] -= sg0
        sgrid[crad+1, 2*crad] -= sg0
        sgrid[crad+1, 2] -= sg0
        sgrid[2, crad+1] -= sg0
    
    sgrid[crad+1, crad+1] = min(sgrid[crad+1, crad+1], 1)
    h = sgrid / np.sum(sgrid)
    
    return h

P = 5  # Радиус диска
PSF = generate_sgrid(P)

print(PSF)
print(PSF.shape)

'''
Реализовать два метода компенсации смазывания изображений
Исследовать влияние неточности задания модели искажения в алгоритм

TODO:
[x] Выбрать изображение
[x] Исказить изображение "смазыванием", без дополнительного зашумления (листинг 2)
[] Компенсировать созданные искажения (листинг 3)
[] Исказить смазанное изображение при помощи шума по варианту
[] Повторить процедуру восстановления (+ сравнение результатов)
[] Взять изображение с расфокусировкой, оценить модель искажения, провести восстановление

Вариант искажения:
%% сферическая расфокусировка радиусом в P пикселей
PSF = fspecial('disk',P);
S = imfilter(input_image, PSF,'replicate');

Вариант шума:
% гауссовский шум (с нормированной дисперсией 0.0001)
distorted_image = imnoise(S,'gaussian',0,0.0001);
% т.о. нормированное СКО равно 0.01 отн. ед., значит без нормировки это дает 2.55 ед. яркости


>> temp
         0         0         0    0.0012    0.0050    0.0063    0.0050    0.0012         0         0         0
         0    0.0000    0.0062    0.0124    0.0127    0.0127    0.0127    0.0124    0.0062    0.0000         0
         0    0.0062    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0062         0
    0.0012    0.0124    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0124    0.0012
    0.0050    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0050
    0.0063    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0063
    0.0050    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0050
    0.0012    0.0124    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0124    0.0012
         0    0.0062    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0127    0.0062         0
         0    0.0000    0.0062    0.0124    0.0127    0.0127    0.0127    0.0124    0.0062    0.0000         0
         0         0         0    0.0012    0.0050    0.0063    0.0050    0.0012         0         0         0

>> 

'''