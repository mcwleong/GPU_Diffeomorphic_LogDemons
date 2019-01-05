close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './gpu_results'

cux = loadbin('ux_cpu.bin');
cuy = loadbin('uy_cpu.bin');
cuz = loadbin('uz_cpu.bin');
gux = loadbin('ux_gpu.bin');
guy = loadbin('uy_gpu.bin');
guz = loadbin('uz_gpu.bin');

compare(cux, gux);
compare(cuy, guy);
compare(cuz, guz);