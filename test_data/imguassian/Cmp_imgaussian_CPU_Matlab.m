close all
clear all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './matlab_results'

uxg = loadbin('uxg.bin');
uyg = loadbin('uyg.bin');
uzg = loadbin('uzg.bin');
uxgm = loadbin('uxgm.dat');
uygm = loadbin('uygm.dat');
uzgm = loadbin('uzgm.dat');

compare(uxg, uxgm);
compare(uyg, uygm);
compare(uzg, uzgm);