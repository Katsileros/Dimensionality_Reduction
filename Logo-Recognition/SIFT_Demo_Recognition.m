%% SIFT Feature extraction demo
clear all; close all; 
% clc;

% Add VLFeat library to path
%addpath('vlfeat-0.9.20/');
% Setup VLFeat
% -> toolbox -> vl_setup;

% im_data = imread('poses/screenshot0.png');
% im_data = single(rgb2gray(im_data));
% 
% im_scene = imread('color0.jpg');
% im_scene = single(rgb2gray(im_scene));
% 
% % imshow(im_data,[]); hold on;
% 
% [fa, da] = vl_sift(im_data) ;
% [fb, db] = vl_sift(im_scene) ;
% [matches, scores] = vl_ubcmatch(da, db) ;


if nargin == 0
    im1 = imread('poses/screenshot0.png') ;
    im2 = imread('poses/screenshot1.png') ;
    
%     im1 = imread('516366872.jpg') ;
%     im2 = imread('1925162720.jpg') ;
end

% make single
im1 = im2single(im1) ;
im2 = im2single(im2) ;

% make grayscale
if size(im1,3) > 1, im1g = rgb2gray(im1) ; else im1g = im1 ; end
if size(im2,3) > 1, im2g = rgb2gray(im2) ; else im2g = im2 ; end

% --------------------------------------------------------------------
%                                                         Cluster the scene
% --------------------------------------------------------------------



% --------------------------------------------------------------------
%                                                         SIFT features and
%                                                         descriptors
% --------------------------------------------------------------------

[f1,d1] = vl_sift(im1g) ;
[f2,d2] = vl_sift(im2g) ;

% --------------------------------------------------------------------
%                                                         LLE    
%                                                         Dimensionality
%                                                         reduction
% --------------------------------------------------------------------
dimRed = 1;
if dimRed
    K = 32;
    d = 36;
    [d1] = lle(double(d1),K,d);
    [d2] = lle(double(d2),K,d);
end
% --------------------------------------------------------------------
%                                                         SIFT matches
% --------------------------------------------------------------------

[matches, scores] = vl_ubcmatch(d1,d2) ;

numMatches = size(matches,2) ;

X1 = f1(1:2,matches(1,:)) ; X1(3,:) = 1 ;
X2 = f2(1:2,matches(2,:)) ; X2(3,:) = 1 ;

% --------------------------------------------------------------------
%                                                         Show matches
% --------------------------------------------------------------------

dh1 = max(size(im2,1)-size(im1,1),0) ;
dh2 = max(size(im1,1)-size(im2,1),0) ;

figure(1) ; clf ;
imagesc([padarray(im1,dh1,'post') padarray(im2,dh2,'post')]) ;
o = size(im1,2) ;
line([f1(1,matches(1,:));f2(1,matches(2,:))+o], ...
     [f1(2,matches(1,:));f2(2,matches(2,:))]) ;
title(sprintf('%d matches', numMatches)) ;
axis image off ;

