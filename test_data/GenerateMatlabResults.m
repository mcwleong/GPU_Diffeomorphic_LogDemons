% This script calls the functions in the MATLAB_ref folder to generate test data

clear all
load Reg_result_10.mat
% 
% %% Generate data for findupdate
% % Data
% saveToBin(F,'test_data\\findupdate\\data\\F.dat');
% saveToBin(Mp,'test_data\\findupdate\\data\\Mp.dat');
% 
% %results
% [uxm uym uzm] = findupdate(F,Mp, 1,1);
% saveToBin(uxm,'test_data\\findupdate\\matlab_results\\uym.dat');  %ux and uy are interchanged because matlab is column-major
% saveToBin(uym,'test_data\\findupdate\\matlab_results\\uxm.dat');
% saveToBin(uzm,'test_data\\findupdate\\matlab_results\\uzm.dat');
% 
% clear uxm uym uzm
% %% Generate data for imgaussian
% %data
% saveToBin(ux,'test_data\\imguassian\\data\\ux.dat');
% saveToBin(uy,'test_data\\imguassian\\data\\uy.dat');
% saveToBin(uz,'test_data\\imguassian\\data\\uz.dat');
% 
% %results
% uxgm = imgaussian(ux, 3);
% uygm = imgaussian(uy, 3);
% uzgm = imgaussian(uz, 3);
% saveToBin(uxgm,'test_data\\imguassian\\matlab_results\\uxgm.dat');
% saveToBin(uygm,'test_data\\imguassian\\matlab_results\\uygm.dat');
% saveToBin(uzgm,'test_data\\imguassian\\matlab_results\\uzgm.dat');
% 
% clear uxgm uygm uzgm
% % Generate data for compose
% data
% saveToBin(ux,'test_data\\compose\\data\\ux.dat'); %ux and uy are interchanged because matlab is column-major
% saveToBin(uy,'test_data\\compose\\data\\uy.dat');
% saveToBin(uz,'test_data\\compose\\data\\uz.dat');
% saveToBin(vx,'test_data\\compose\\data\\vx.dat');
% saveToBin(vy,'test_data\\compose\\data\\vy.dat');
% saveToBin(vz,'test_data\\compose\\data\\vz.dat');
% 
% %results 
% [vxm vym vzm] = compose(vx, vy, vz, ux, uy, uz);
% 
% saveToBin(vxm,'test_data\\compose\\matlab_results\\vxm.dat');
% saveToBin(vym,'test_data\\compose\\matlab_results\\vym.dat');
% saveToBin(vzm,'test_data\\compose\\matlab_results\\vzm.dat');
% 
% clear vxm vym vzm
% 
% %% Generdate data for expfield
% %Smooth
% vxg = imgaussian(vx, 3);
% vyg = imgaussian(vy, 3);
% vzg = imgaussian(vz, 3);
% 
% %data
% saveToBin(vxg,'test_data\\expfield\\data\\vxg.dat');
% saveToBin(vyg,'test_data\\expfield\\data\\vyg.dat');
% saveToBin(vzg,'test_data\\expfield\\data\\vzg.dat');
% 
% %results
% % [sxm sym szm] = expfield(vxg, vyg, vzg);
% % 
% % saveToBin(sxm,'test_data\\expfield\\matlab_results\\sxm.dat');
% % saveToBin(sym,'test_data\\expfield\\matlab_results\\sym.dat');
% % saveToBin(szm,'test_data\\expfield\\matlab_results\\szm.dat');
% % 
% % clear vxg vyg vzg sxm sym szm
% % 
% 
% %% generate data for iminterpolate
% %data
% saveToBin(sx,'test_data\\iminterpolate\\data\\sx.dat');
% saveToBin(sy,'test_data\\iminterpolate\\data\\sy.dat');
% saveToBin(sz,'test_data\\iminterpolate\\data\\sz.dat');
% saveToBin(M,'test_data\\iminterpolate\\data\\M.dat');
% 
% Mpm = iminterpolate(M, sx, sy, sz);
% 
% %result
% saveToBin(Mpm,'test_data\\iminterpolate\\matlab_results\\Mpm.dat');
% 
% clear Mpm

%% generate data for energy
%data
saveToBin(sx,'test_data\\energy\\data\\sx.dat');
saveToBin(sy,'test_data\\energy\\data\\sy.dat');
saveToBin(sz,'test_data\\energy\\data\\sz.dat');
saveToBin(Mp,'test_data\\energy\\data\\Mp.dat');
saveToBin(F,'test_data\\energy\\data\\F.dat');

jacm = jacobian(sx,sy,sz);
saveToBin(jacm, 'test_data\\energy\\matlab_results\\jacm.bin');
