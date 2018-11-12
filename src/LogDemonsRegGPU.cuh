#pragma once
#include "LogDemonsReg.cuh"
#include "LogDemonsRegGPUKernels.cuh"

// cuda includes
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// thrust include
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#ifdef __CUDACC__ //get rid of the inteliSense errors...
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


class LogDemonsRegGPU : public LogDemonsReg
{
public:
	LogDemonsRegGPU(){};
	LogDemonsRegGPU(float *F, float *M, unsigned int dimension[3]):LogDemonsReg(F, M, dimension){
		cudaDeviceReset();
	}
	void setDeviceID(int i){ deviceID = i; };
	void printDeviceProp(int deviceID);
	void getCudaError(const char* s);
	void initialize();
	void Register();
	
private:
	int deviceID = 0;

	cudaTextureObject_t initGaussValue(float* &h_gaussianVal, float* &d_gaussVal, float sigma, int &radius);
	cudaTextureObject_t CreateTextureObject(float *d_I, cudaArray *d_cuArr);

	inline int maxdim(unsigned int dim[3]){
		return dim[0] > dim[1] ? (dim[0] > dim[2] ? dim[0] : dim[2]) : (dim[1] > dim[2] ? dim[1] : dim[2]);
	}

	// Log demons function
	void findupdate();
	void imgaussian(float* d_fx, float* d_fy, float* d_fz, cudaTextureObject_t tex, int radius);
	void compose();
	void self_compose();
	void expfield();
	void iminterpolate();
	void interpolate(cudaTextureObject_t tex_I, float* d_sx, float* d_sy, float* d_sz, float* Ip);
	float energy();

	void gradient(float* d_I, float* d_fxg, float* d_fyg, float* d_fzg);
	float thrustFindMaxElement(float* d_f);
	
	// device memory pointers
	float *d_fixed = 0, *d_moving = 0, *d_deformedMoving = 0;
	float *d_ux = 0, *d_uy = 0, *d_uz = 0, *d_vx = 0, *d_vy = 0, *d_vz = 0, *d_sx = 0, *d_sy = 0, *d_sz = 0, *d_normg2 = 0, *d_det_J = 0;
	float *d_uxf = 0, *d_uyf = 0, *d_uzf = 0;
	float *d_en;
	int radius_f, radius_d;
	float* gaussian_f = 0, *gaussian_d = 0;
	float *d_gaussian_f = 0, *d_gaussian_d = 0;
	dim3 d3_dim;
	
	cudaTextureObject_t tex_gauss_f, tex_gauss_d;
	cudaArray *d_cuArr_mov;
	cudaArray *d_cuArr_vx, *d_cuArr_vy, *d_cuArr_vz;
	cudaTextureObject_t tex_mov;
	cudaTextureObject_t tex_vx, tex_vy, tex_vz;
};
