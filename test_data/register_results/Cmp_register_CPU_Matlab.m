close all
clear all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './matlab_results'

Mp = loadbin('Mp_10.bin');
Mpm = loadbin('Mpm_10.bin');
compare(Mp, Mpm);