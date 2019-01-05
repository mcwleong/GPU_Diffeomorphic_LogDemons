clear all
close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './gpu_results'
addpath './data'
ux = loadbin('ux.dat');
uy = loadbin('uy.dat');
uz = loadbin('uz.dat');
cuxg = loadbin('uxg.bin');
cuyg = loadbin('uyg.bin');
cuzg = loadbin('uzg.bin');
guxg = loadbin('uxg_gpu.bin');
guyg = loadbin('uyg_gpu.bin');
guzg = loadbin('uzg_gpu.bin');

compare(cuxg,guxg)
%compare(cuyg,guyg)
%compare(cuzg,guzg)

% scux = sign(cux);
% sgux = sign(gux);
% compare(scux, sgux);