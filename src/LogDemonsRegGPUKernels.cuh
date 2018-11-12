#include <iostream>
//cuda includes
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

enum AddSubtractCoordinateImage{
	addCoordinate = 0,
	substractCoordinate
};

__global__ void gradientKernel(float* I, float* fxg, float* fyg, float* fzg, dim3 paddim);

__global__ void updateKernel(float* F, float* Mp, float* ux, float* uy, float* uz, float* uxf, float* uyf, float* uzf, float alpha2, dim3 paddim);

__global__ void gaussianKernel(float* I, cudaTextureObject_t tex, dim3 dim, int radius, int pass);

__global__ void coordinateImageKernel(float *vx, float* vy, float* vz, dim3 dim, AddSubtractCoordinateImage op);

__global__ void interpolateImageKernel(cudaTextureObject_t tex, float* sx, float* sy, float* sz, float* Ip, dim3 dim);

__global__ void normalizeVectorKernel(float* fx, float* fy, float* fz, float* norm2, unsigned int len);

__global__ void scaleKernel(float* vx, float* vy, float* vz, float* sx, float* sy, float* sz, float scale, unsigned int len);

__global__ void energyKernel(float* fixed, float* Mp, float* sx, float* sy, float*sz, float* d_en, float reg_weight, dim3 dim);

__device__ unsigned int pos(dim3 pos, dim3 dim){
	return pos.x + pos.y*dim.x + pos.z*dim.x * dim.y;
}

__device__ int sign(float val) {
	return (float(0) < val) - (val<float(0));
}