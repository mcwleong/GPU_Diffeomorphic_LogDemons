Functions in MATLAB_IO are neccessary to communicate with the C++/CUDA implementation.

Format: dimensions of the image in x, y, z directions stored in short int format; Image data stored in float32 format follows immediately afterwards.
[short] [short] [short] [float] ... [float]

SaveToBin.m
A simple script that convert any MATLAB variable into a bin file that store the data as well as the dimension of the variable. Combine this simple "saveToBin" tool with the MATLAB dicomread function to efficiently generate dataset compatible to this C++/CUDA implementation

Loadbin.m
A simple script that read the .bin file from the C++/CUDA implementation as a MATLAB variable.
