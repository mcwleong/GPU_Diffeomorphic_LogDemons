#include "Registration.h"
#include "registrationKernels.cuh"

//CUDA include
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//thrust include
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

//other include
//#define __CUDACC__


#ifdef __CUDACC__ //get rid of the inteliSense errors...
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

using namespace std;
using namespace std::chrono;

class RegistrationGPU : public Registration
{
public:
	RegistrationGPU(float *Fixed, float *Moving, unsigned int dimension[3]);
	~RegistrationGPU();
	
	virtual void Register();  //Overwritten public function for GPU
	void SetDeviceID(int i);
	void printDevProp(int deviceID);

private:
	void getCudaError(const char* s);
	void initialize();
	void padImage();
	void findupdate(float* d_F, float* d_Mp, float4* d_um);
	void gradient(float* d_I, float4* d_g);
	void imgaussian(float4* d_u, cudaTextureObject_t mask, int mask_size);
	void compose(float4* d_v, float4* d_u);
	void compose(float4* d_s);

	float FindMaxU(float4 *d_u);
	void expfield(float4 *d_u, float4 *d_v);
	void iminterpolate(float * d_Mp, float4 * d_s);
	float energy(float* d_f, float* d_Mp, float4 *d_s);
	void saveResult(float* d_Mp, float4* d_s);


	//debug use
	float cpuEnergy(float* d_f, float* d_Mp, float4 *d_s);

	void makefloat(float4* h_f4, float* h_f, int col);
	void Savetobin(float4* f4, int col, std::string filename);
	void Savetobin(float* f, std::string filename);

	template <typename T> 
	cudaTextureObject_t CreateTextureObject(T* d_u);


	float FindMaxSx(float4 *d_s);

	int deviceID = 0;

	bool GPUinitialized = false;
	float *d_fixed, *d_moving, *d_deformedMoving, *d_e, *d_uw;
	float4 *d_u, *d_v, *d_s, *d_uf;
	dim3 padDim;

	//Gaussian values stored in texture memory
	int mask_width_f, mask_width_d;
	float *gaussianMask_f, *gaussianMask_d;
	cudaResourceDesc Res_maskf, Res_maskd;
	cudaTextureDesc Tex_maskf, Tex_maskd;
	cudaTextureObject_t d_texf, d_texd;
	cudaTextureObject_t d_tex_M;
	cudaArray *d_cuArr_float, *d_cuArr_float4;
	float *d_maskf, *d_maskd;

	float* e_cpu;
};
