close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './gpu_results'

cvx = loadbin('vx.bin');
cvy = loadbin('vy.bin');
cvz = loadbin('vz.bin');
gvx = loadbin('vx_gpu.bin');
gvy = loadbin('vy_gpu.bin');
gvz = loadbin('vz_gpu.bin');

compare(cux,gux)
compare(cuy,guy)
compare(cuz,guz)

% scux = sign(cux);
% sgux = sign(gux);
% compare(scux, sgux);