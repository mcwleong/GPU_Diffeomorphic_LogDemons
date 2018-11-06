#include <stdio.h>
#include <math.h>
//CUDA include
#include "cuda.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
//#define THRESHOLD 12.0f  //threshold filter at padImageKernel


__global__ void padImageKernel(float* fixedsrc, float* movingsrc, float* fixed, float* moving, float* deformedMoving, const dim3 dim, const dim3 padDim, const dim3 marginSize);

__device__ unsigned int pos(dim3 pos, int paddedlen[3]);

__global__ void gradientKernel(float* image, float4* g, dim3 dim);

__device__ int sign(float val);

__global__ void updateKernel(float* F, float* Mp, float4* u, float4* uf, float alpha2, dim3 dim);

__global__ void gaussianKernel(float4* u, cudaTextureObject_t mask, int kernel_radius, dim3 dim, int dir);

__global__ void coordinateImageKernel(float4* u, dim3 dim);

__global__ void composeKernel(float4* v, cudaTextureObject_t tex_u, dim3 dim);

__global__ void selfComposeKernel(float4* s, cudaTextureObject_t tex_s, dim3 dim);

__global__ void interpolateImageKernel(float* Mp, float4* s, cudaTextureObject_t tex_M, dim3 dim);

__global__ void u2uw(float4* u, float* uw, unsigned int len);

__global__ void scaleKernel(float4* s, float4* u, float scale, unsigned int paddedlen);

__global__ void energyKernel(const float * F, const float* Mp, const float4* s, float* e, float reg_weight, dim3 dim);