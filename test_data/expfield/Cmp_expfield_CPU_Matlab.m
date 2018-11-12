clear all
close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './matlab_results'

sx = loadbin('sx.bin');
sy = loadbin('sy.bin');
sz = loadbin('sz.bin');
sxm = loadbin('sxm.dat');
sym = loadbin('sym.dat');
szm = loadbin('szm.dat');

compare(sx, sxm);
