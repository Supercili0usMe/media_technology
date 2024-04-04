import cv2
import matplotlib.pyplot as plt
import functions as f

#------------------ Основная функция -------------------
city = cv2.imread('7/st_petersburg.jpg', cv2.IMREAD_COLOR)
f.image_spectre(city, True)

# Применение фильтра Лапласа
c = 0.5
remove_blured_city = f.laplas_filter(city, c)
f.image_spectre(remove_blured_city, True)
f.compare2imgs(city, remove_blured_city, 'Исходная картинка', 
             f'Фильтр Лапласа ({c = })', 'Сравнение результата повышения четкости')

# Размыливание картинки
intense = 5
blured_city, PSF = f.bluring_image(city, intense)
f.image_spectre(blured_city, True)
f.compare2imgs(city, blured_city, 'Исходная картинка', 
             f'Размыливание картинки ({intense = })', 'Сравнение результата понижения четкости')

# Восстановления картинки Тихоновым - не удалось

# Восстановление картинки Винером - ч/б
noise_level = 0.01
unblured_city = f.wiener(blured_city, PSF, noise_level)
f.image_spectre(unblured_city, True)
f.compare2imgs(blured_city, unblured_city, 'Смазанная картинка', 
              f'Восстановленная картинка ', 'Сравнение результата понижения четкости')

# Добавление шума на изображение
mean, var = 0, 0.001
noisy_city = f.gaussian_noise(blured_city, mean, var)
f.image_spectre(noisy_city, True)
f.compare2imgs(blured_city, noisy_city, 'Смазанная картинка', 
               'Зашумленная картинка', 'Сравнение результата добавления гауссовского шума')

# Восстановление четкости
bread = cv2.imread('7/IMG_0108.jpg', cv2.IMREAD_COLOR)
c = 3
unblured_bread = f.laplas_filter(bread, c)
f.compare2imgs(bread, unblured_bread, 'Исходная картинка', 
              f'Фильтр Лапласа ({c = })', 'Сравнение результата повышения четкости')

plt.show()

'''
Реализовать два метода компенсации смазывания изображений
Исследовать влияние неточности задания модели искажения в алгоритм

TODO:
[+] Выбрать изображение
[+] Исказить изображение "смазыванием", без дополнительного зашумления (листинг 2)
[+] Компенсировать созданные искажения (листинг 3)
    [-] Переписать deconvreg()
    [+-] restoration.wiener()
[+] Исказить смазанное изображение при помощи шума по варианту
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

'''