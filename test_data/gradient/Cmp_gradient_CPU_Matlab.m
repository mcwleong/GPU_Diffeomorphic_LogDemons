close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './matlab_results'

ux = loadbin('ux.bin');
uy = loadbin('uy.bin');
uz = loadbin('uz.bin');
uxm = loadbin('uxm.dat');
uym = loadbin('uym.dat');
uzm = loadbin('uzm.dat');

compare(ux, uxm);

compare(uy, uym);
compare(uz, uzm);