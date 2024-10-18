
%%% Dawa Chyophel Lepcha, Bhawna Goyal, Ayush Dogra, and Shui?Hua Wang. "An efficient medical image super resolution 
%%% based on piecewise linear regression strategy using domain transform filtering." 
%%% Concurrency and Computation: Practice and Experience 34, no. 20 (2022): e6644. ( https://doi.org/10.1002/cpe.6644)

clc;
clear;
close all;

scale = 2 ;
load(['parameters1\parameter_' num2str(scale)]);
load(['parameters1\dt_' num2str(scale)]);
image=im2double(imread('Test\Set1\c03_1.bmp'));

%% Weighted least squares(WLS) filter optimization framework

cform = makecform('srgb2lab');
lab = applycform(image, cform);
L = lab(:,:,1);
tic
L0 = wlsFilter(L, 0.125, 1.2);
L1 = wlsFilter(L, 0.5, 1.2);
toc

% Coarse
val0 = 30;
val1 = 1;
val2 = 1;
exposure = 1.0;
saturation = 1.0;
gamma = 1.0;

coarse = tonemapLAB(lab, L0, L1,val0,val1,val2,exposure,gamma,saturation);

%% RF  Domain transform recursive edge-preserving filter.
sigma_s = 5;
sigma_r = 0.4;

F_rf = RF(coarse, sigma_s, sigma_r);

%% Piecewise linear regression via hadamard transform
image  = modcrop(F_rf,scale);

h = fspecial('gaussian', 5, 1.6);  % Gaussian filter
image_gauss = imfilter( image, h);

H_15 = [2 8 3 12 10 1 4 11 14 6 9 15 7 13 5];   % sequence

sz1 = size(image);

if(size(sz1,2)==2)  
    imageL = imresize(image_gauss,1/scale,'bicubic');
    imageB = imresize(imageL,scale,'bicubic');
else
    image_ycbcr = rgb2ycbcr(image_gauss);
    
    image_y  = im2double(image_ycbcr(:,:,1));
    image_cb = im2double(image_ycbcr(:,:,2));
    image_cr = im2double(image_ycbcr(:,:,3));
    
    imageL    = imresize(image_y,1/scale,'bicubic');
    imageL_cb = imresize(image_cb,1/scale,'bicubic');
    imageL_cr = imresize(image_cr,1/scale,'bicubic');
    
    imageB = zeros(size(image_ycbcr));
    imageB(:,:,1) = imresize(imageL,scale,'bicubic');
    imageB(:,:,2) = imresize(imageL_cb,scale,'bicubic');
    imageB(:,:,3) = imresize(imageL_cr,scale,'bicubic');
    
    imageH_rec = zeros(size(image_ycbcr));
    imageH_rec(:,:,2) = imageB(:,:,2);
    imageH_rec(:,:,3) = imageB(:,:,3);
end

H_16=hadamard( 16 ); % the Hadamard matrix
H_16(:,1) =[]; 

sz = size(imageL);
imagepadding = zeros(sz(1)+2,sz(2)+2);  % image padding
imagepadding(2:end-1,2:end-1) = imageL;

offset = floor( scale / 2 );
    
startt = tic;  
[imageH]= SR_2_Hadamard( imagepadding, parameters, dt, H_16 );
toc(startt);

if(size(sz1,2)==2)
    imageH_rec = imageH;
else  
    imageH_rec(:,:,1) = imageH;
    imageB = ycbcr2rgb( imageB );
    imageH_rec = ycbcr2rgb( imageH_rec );
end

% Display 
figure('NumberTitle', 'off', 'Name', 'Our');
imshow(imageH_rec,'Border','tight');
