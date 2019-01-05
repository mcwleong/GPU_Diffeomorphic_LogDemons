clear all

addpath('..\Matlab_ref');

% original data
load Reg_result_10.mat
 
%% Generate data for findupdate
% Data
saveToBin(F,'.\\findupdate\\data\\F.dat');
saveToBin(Mp,'.\\findupdate\\data\\Mp.dat');

%results
[uxm uym uzm] = findupdate(F,Mp, 1,1);
saveToBin(uxm,'.\\findupdate\\matlab_results\\uym.dat');  %ux and uy are interchanged because matlab is column-major
saveToBin(uym,'.\\findupdate\\matlab_results\\uxm.dat');
saveToBin(uzm,'.\\findupdate\\matlab_results\\uzm.dat');

clear uxm uym uzm
%% Generate data for imgaussian
%data
saveToBin(ux,'.\\imguassian\\data\\ux.dat');
saveToBin(uy,'.\\imguassian\\data\\uy.dat');
saveToBin(uz,'.\\imguassian\\data\\uz.dat');

%results
uxgm = imgaussian(ux, 3);
uygm = imgaussian(uy, 3);
uzgm = imgaussian(uz, 3);
saveToBin(uxgm,'.\\imguassian\\matlab_results\\uxgm.dat');
saveToBin(uygm,'.\\imguassian\\matlab_results\\uygm.dat');
saveToBin(uzgm,'.\\imguassian\\matlab_results\\uzgm.dat');

clear uxgm uygm uzgm
% Generate data for compose
%data
saveToBin(ux,'.\\compose\\data\\ux.dat'); %ux and uy are interchanged because matlab is column-major
saveToBin(uy,'.\\compose\\data\\uy.dat');
saveToBin(uz,'.\\compose\\data\\uz.dat');
saveToBin(vx,'.\\compose\\data\\vx.dat');
saveToBin(vy,'.\\compose\\data\\vy.dat');
saveToBin(vz,'.\\compose\\data\\vz.dat');

%results 
[vxm vym vzm] = compose(vx, vy, vz, ux, uy, uz);

saveToBin(vxm,'.\\compose\\matlab_results\\vxm.dat');
saveToBin(vym,'.\\compose\\matlab_results\\vym.dat');
saveToBin(vzm,'.\\compose\\matlab_results\\vzm.dat');

clear vxm vym vzm

%% Generdate data for expfield
%Smooth
vxg = imgaussian(vx, 3);
vyg = imgaussian(vy, 3);
vzg = imgaussian(vz, 3);

%data
saveToBin(vxg,'.\\expfield\\data\\vxg.dat');
saveToBin(vyg,'.\\expfield\\data\\vyg.dat');
saveToBin(vzg,'.\\expfield\\data\\vzg.dat');

%results
% [sxm sym szm] = expfield(vxg, vyg, vzg);
% 
% saveToBin(sxm,'.\\expfield\\matlab_results\\sxm.dat');
% saveToBin(sym,'.\\expfield\\matlab_results\\sym.dat');
% saveToBin(szm,'.\\expfield\\matlab_results\\szm.dat');
% 
% clear vxg vyg vzg sxm sym szm
% 

%% generate data for iminterpolate
%data
saveToBin(sx,'.\\iminterpolate\\data\\sx.dat');
saveToBin(sy,'.\\iminterpolate\\data\\sy.dat');
saveToBin(sz,'.\\iminterpolate\\data\\sz.dat');
saveToBin(M,'.\\iminterpolate\\data\\M.dat');

Mpm = iminterpolate(M, sx, sy, sz);

%result
saveToBin(Mpm,'.\\iminterpolate\\matlab_results\\Mpm.dat');

clear Mpm

%% generate data for energy
%data
saveToBin(sx,'.\\energy\\data\\sx.dat');
saveToBin(sy,'.\\energy\\data\\sy.dat');
saveToBin(sz,'.\\energy\\data\\sz.dat');
saveToBin(Mp,'.\\energy\\data\\Mp.dat');
saveToBin(F,'.\\energy\\data\\F.dat');

jacm = jacobian(sx,sy,sz);
saveToBin(jacm, '.\\energy\\matlab_results\\jacm.bin');
