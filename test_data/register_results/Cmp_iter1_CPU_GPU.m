
close all
addpath '../../Matlab_IO'
addpath './cpu_results/iter1'
addpath './gpu_results/iter1'

cux = loadbin('ux.bin');
cuy = loadbin('uy.bin');
cuz = loadbin('uz.bin');
cuxg = loadbin('uxg.bin');
cuyg = loadbin('uyg.bin');
cuzg = loadbin('uzg.bin');
cvx = loadbin('vx.bin');
cvy = loadbin('vy.bin');
cvz = loadbin('vz.bin');
cvxg = loadbin('vxg.bin');
cvyg = loadbin('vyg.bin');
cvzg = loadbin('vzg.bin');
csx = loadbin('sx.bin');
csy = loadbin('sy.bin');
csz = loadbin('sz.bin');
cmp = loadbin('Mp.bin');

gux = loadbin('ux_gpu.bin');
guy = loadbin('uy_gpu.bin');
guz = loadbin('uz_gpu.bin');
guxg = loadbin('uxg_gpu.bin');
guyg = loadbin('uyg_gpu.bin');
guzg = loadbin('uzg_gpu.bin');
gvx = loadbin('vx_gpu.bin');
gvy = loadbin('vy_gpu.bin');
gvz = loadbin('vz_gpu.bin');
gvxg = loadbin('vxg_gpu.bin');
gvyg = loadbin('vyg_gpu.bin');
gvzg = loadbin('vzg_gpu.bin');
gsx = loadbin('sx_gpu.bin');
gsy = loadbin('sy_gpu.bin');
gsz = loadbin('sz_gpu.bin');
gmp = loadbin('Mp_gpu.bin');
%findupdate - checked
%  compare(cux,gux)
%  compare(cuy,guy)
%  compare(cuz,guz)

%imgaussian 0 - checked with speckle defect
% compare(cuxg,guxg)
% compare(cuyg,guyg)
% compare(cuzg,guzg)

%compose - checked
% compare(cvx, gvx);
% compare(cvy, gvy);
% compare(cvz, gvz);

% imgaussian 1 - checked
%  compare(cvxg, gvxg);
%  compare(cvyg, gvyg);
%  compare(cvzg, gvzg);

%expfield
% compare(csx,gsx)
% compare(csy,gsy)
% compare(csz,gsz)

% iminterpolate
compare(cmp,gmp);