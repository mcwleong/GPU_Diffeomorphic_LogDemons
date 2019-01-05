#pragma once

#include <iostream>
//cuda includes
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum AddSubtractCoordinateImage{
	addCoordinate = 0,
	substractCoordinate
};

__global__ void gradientKernel(float* I, float* fg, int dir, dim3 dim);

__global__ void updateKernel(float* F, float* Mp, float* ux, float* uy, float* uz, float* uxf, float* uyf, float* uzf, float alpha2, unsigned int len);

__global__ void gaussianKernel(float* I, cudaTextureObject_t tex, dim3 dim, int radius, int pass);

__global__ void coordinateImageKernel(float *vx, float* vy, float* vz, dim3 dim, AddSubtractCoordinateImage op);

__global__ void interpolateImageKernel(cudaTextureObject_t tex, float* sx, float* sy, float* sz, float* Ip, dim3 dim);

__global__ void normalizeVectorKernel(float* fx, float* fy, float* fz, float* norm2, unsigned int len);

__global__ void scaleKernel(float* vx, float* vy, float* vz, float* sx, float* sy, float* sz, float scale, unsigned int len);

__global__ void jacobianKernel(float* sx, float* sy, float*sz, float* jac2, dim3 dim);

__global__ void energyKernel(float* Mp, float* F, float* jac2, float* en, float reg_weight, unsigned int len);


__device__ inline unsigned int gpos(dim3 pos, dim3 dim){
	return pos.x + pos.y*dim.x + pos.z*dim.x * dim.y;
}

__device__ inline int sgn(float val) {
	return (float(0) < val) - (val<float(0));
}

__device__ inline bool operator<(dim3 a, dim3 b){
	return (a.x < b.x) && (a.y < b.y) && (a.z < b.z);
}