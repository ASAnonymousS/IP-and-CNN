clear all
close all
clc


img = rgb2gray(imread("Image Location"));

figure(1)
imshow(img)
imhist(img)

[X,Y]=size(img);

BI=zeros(X,Y);
for i=1:X
    for j=1:Y
        % Process each pixel (example: thresholding)
        if img(i, j) > 110
            BI(i, j) = 255; % Set to white
        else
            BI(i, j) = 0;   % Set to black
        end
    end
end

figure(2)

imshow(BI)
