close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './gpu_results'

csx = loadbin('sx.bin');
csy = loadbin('sy.bin');
csz = loadbin('sz.bin');
gsx = loadbin('sx_gpu.bin');
gsy = loadbin('sy_gpu.bin');
gsz = loadbin('sz_gpu.bin');

compare(csx,gsx)
compare(csy,gsy)
compare(csz,gsz)
