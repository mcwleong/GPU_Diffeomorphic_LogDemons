close all
clear all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './matlab_results'

Mp = loadbin('Mp_10.bin');
Mpm = loadbin('Mpm_10.bin');
%compare(Mp, Mpm);

diff = Mp - Mpm;
diff = abs(diff)*100./max(Mp, Mpm);
colorbar

imagesc(getslice(diff,70));
colormap jet

% %colormap(gray);
% figure
% Mp1 = getslice(Mp,70);
% Mpm1 = getslice(Mpm,70);
% imshowpair(Mp1, Mpm1, 'diff');
% %Mp1 = getslice(Mp,70);
% %Mpm1 = getslice(Mpm,70);