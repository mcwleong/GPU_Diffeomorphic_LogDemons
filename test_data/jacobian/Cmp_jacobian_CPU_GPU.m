close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './gpu_results'

cjac = loadbin('jac.bin');

gjac = loadbin('jac_gpu.bin');
 
compare(cjac, gjac);