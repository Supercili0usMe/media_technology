import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, color
from skimage.util import random_noise

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
    plt.title('Изображение')
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

# Функция создания маски для сферической расфокусировки
def fspecial_disk_selfmade(p2):
    rad = p2
    crad = int(np.ceil(rad - 0.5))
    
    x_min, x_max = -crad, crad
    y_min, y_max = -crad, crad
    
    x, y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    
    maxxy = np.maximum(np.abs(x), np.abs(y))
    minxy = np.minimum(np.abs(x), np.abs(y))
    
    m1 = np.where(rad**2 < (maxxy + 0.5)**2 + (minxy - 0.5)**2, (minxy - 0.5), np.sqrt(np.abs(rad**2 - (maxxy + 0.5)**2)))
    m2 = np.where(rad**2 > (maxxy - 0.5)**2 + (minxy + 0.5)**2, (minxy + 0.5), np.sqrt(np.abs(rad**2 - (maxxy - 0.5)**2)))

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

# Функция размыливания картинки
def bluring_image(img, radius: int):
    PSF = fspecial_disk_selfmade(radius)
    distorted_image = cv2.filter2D(img, -1, PSF)
    return distorted_image, PSF

# Функция восстановления методом Винера
def wiener(img, psf, noise_level):
    img = color.rgb2gray(img)
    J = restoration.wiener(img, psf, noise_level)
    J = color.gray2rgb(J)
    max_J = np.max(J)
    output = np.uint8((J / max_J) * 255)
    return output

# Функция добавления гауссовского шума
def gaussian_noise(img, mean=0, var=0.01):
    noisy_image = random_noise(img, mode="gaussian", mean=mean, var=var)
    max_J = np.max(noisy_image)
    output = np.uint8((noisy_image / max_J) * 255)
    return output