
% Функция построения изображения + спектра
function image_spectre(img)
    figure('Position', [100, 100, 1200, 600]);
    sgtitle('Картинка и её спектр');

    subplot(1, 2, 1);
    imshow(img);
    title('Исходное изображение');
    axis off;

    subplot(1, 2, 2);
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    img = double(img);
    S = fftshift(fft2(img));
    A = abs(S);
    A_max = max(A(:));
    eps = A_max * 10^(-6);
    A_dB = 20 * log10(A + eps);
    imagesc(A_dB);
    title('Спектр изображения');
    colorbar;
end

% Функция сравнения 2 картинок
function compare2imgs(img1, img2, title1, title2, main_title)
    figure('Position', [100, 100, 1200, 600]);
    sgtitle(main_title);

    subplot(1, 2, 1);
    imshow(img1);
    title(title1);
    axis off;

    subplot(1, 2, 2);
    imshow(img2);
    title(title2);
    axis off;
end

% Функция размыливания картинки
function [S, PSF] = bluring_image(img, radius)
    PSF = fspecial('disk', radius);
    S = imfilter(img, PSF, 'replicate');
end

% Функция восстановления резкости изображения
function J = deconvolution(img, PSF, noise_level, method)
    img = edgetaper(img, PSF);
    switch method
        case "Tihonov"
            J = deconvreg(img, PSF, [], noise_level);
        case "Wiener"
            J = deconvwnr(img, PSF, noise_level);
    end
end

% Основная часть работы
city = imread('7/st_petersburg.jpg');

% Замыливание изображения
intense = 5; 
[blured_city, PSF] = bluring_image(city, intense);
image_spectre(blured_city);
compare2imgs(city, blured_city, 'Исходная картинка', 'Размыливание картинки', "Сравнение результата понижения четкости");

% Восстановление изображения 
noise_power = 0.01;
unblured_city = deconvolution(blured_city, PSF, noise_power, 'Wiener');
image_spectre(unblured_city);
compare2imgs(blured_city, unblured_city, 'Размыленная картинка', 'Восстановленная картинка', "Сравнение результата восстановления четкости");

