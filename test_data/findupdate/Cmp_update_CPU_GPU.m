close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './gpu_results'

cux = loadbin('ux.bin');
cuy = loadbin('uy.bin');
cuz = loadbin('uz.bin');
gux = loadbin('ux_gpu.bin');
guy = loadbin('uy_gpu.bin');
guz = loadbin('uz_gpu.bin');

compare(cux,gux)
compare(cuy,guy)
compare(cuz,guz)

% scux = sign(cux);
% sgux = sign(gux);
% compare(scux, sgux);