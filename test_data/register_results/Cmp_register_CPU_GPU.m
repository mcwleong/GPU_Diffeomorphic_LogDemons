close all
clear all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './gpu_results'

% cmp1 = loadbin('Mp_1.bin');
% gmp1 = loadbin('Mp_1_gpu.bin');
% compare(cmp1, gmp1);
% 
% cmp2 = loadbin('Mp_2.bin');
% gmp2 = loadbin('Mp_2_gpu.bin');
% compare(cmp2, gmp2);
% 
% cmp3 = loadbin('Mp_3.bin');
% gmp3 = loadbin('Mp_3_gpu.bin');
% compare(cmp3, gmp3);

cmp10 = loadbin('Mp_3.bin');
gmp10 = loadbin('Mp_3_gpu.bin');
compare(cmp10, gmp10, 70);