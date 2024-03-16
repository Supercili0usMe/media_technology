landscape = imread("1.jpg");
kernels = [2, 4, 6, 8, 32, 64];
results = cell(1, 6);
for i = 1:6
    disp(['Расчет при размере ядра: ', num2str(kernels(i))]);
    results{i} = image_adapthisteq(landscape, [kernels(i), kernels(i)], 0.02, false);
end

compare6image(results, kernels);

function compare6image(imgs, params)
    fig1 = figure('Name', 'Изображения', 'Position', [100, 100, 1400, 700]);
    for i = 1:6
        subplot(2, 3, i, 'Parent', fig1);
        imshow(imgs{i});
        title(['Ядро: ', num2str(params(i))]);
        axis off;
    end

    fig2 = figure('Name', 'гистограммы яркости', 'Position', [100, 100, 1400, 700]);
    for i = 1:6
        subplot(2, 3, i, 'Parent', fig2);
        imhist(im2gray(imgs{i}), 256);
        title(['Ядро: ', num2str(params(i))]);
    end
end

function J = image_adapthisteq(img, kernel_size, clip_limit, convert2lab)
    if convert2lab
        LAB = rgb2lab(img);
        L = LAB(:,:,1) / 100;
        L = adapthisteq(L, 'ClipLimit', clip_limit, 'NumTiles', kernel_size);
        LAB(:,:,1) = L * 100;
        J = lab2rgb(LAB);
    else
        gray_img = rgb2gray(img);
        J = adapthisteq(gray_img, 'ClipLimit', clip_limit, 'NumTiles', kernel_size);
    end
end