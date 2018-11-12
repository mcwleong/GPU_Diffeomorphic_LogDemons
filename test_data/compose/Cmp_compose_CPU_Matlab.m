clear all
close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './matlab_results'

vx = loadbin('vx.bin');
vy = loadbin('vy.bin');
vz = loadbin('vz.bin');
vxm = loadbin('vxm.dat');
vym = loadbin('vym.dat');
vzm = loadbin('vzm.dat');

compare(vx, vxm);
