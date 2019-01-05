close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './gpu_results'
cmp = loadbin('Mp.bin');
gmp = loadbin('Mp_gpu.bin');

compare(cmp, gmp)

% scux = sign(cux);
% sgux = sign(gux);
% compare(scux, sgux);