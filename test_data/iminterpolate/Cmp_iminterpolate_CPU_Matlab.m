clear all
close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './matlab_results'

Mp = loadbin('Mp.bin');
Mpm = loadbin('Mpm.dat');

compare(Mp, Mpm);
