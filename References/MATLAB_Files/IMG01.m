close all
clc

img = rgb2gray(imread("Image Location"));

figure(1)
imshow(img)
M=max(max(img));
N=min(min(img));
[X, Y]=size(img);
figure(2)
imhist(img)

img2 = rgb2gray(img);
figure;
imshow(img2);

filter1 = [-1,-1,-1;-1,8,-1;-1,-1,-1];
img3 = imfilter(img,filter1);
figure(3);
imshow(img3);

filter2= [-1 0 1; -1 0 1; -1 0 1];
img4 = imfilter(img, filter2);
figure;
imshow(img4);

filter3 = [-1 -1 -1; 0 0 0; 1 1 1];
img5= imfilter(img, filter3);
figure;
imshow(img5);

Averaging Filter
filter4=[1 1 1 ;1 1 1;1 1 1]*(1/9);
img6 = imfilter(img, filter4);
figure;
imshow(img6);


figure
imhist(img);

figure
imhist(img2);



Test

A=[1 2 3 4 5 4 3 7 9 1 9];
for i=1:11
    if A(i)>5
        A(i)=10;
    else
        A(i)=0;
    end
end
