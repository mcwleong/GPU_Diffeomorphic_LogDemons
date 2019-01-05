close all
addpath '../../Matlab_IO'
addpath './cpu_results'
addpath './matlab_results'

jacm = loadbin('jacm.bin');
jac = loadbin('jac.bin');


compare(jac, jacm);