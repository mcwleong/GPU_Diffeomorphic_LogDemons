#include "registrationKernels.cuh"

__device__ unsigned int pos(dim3 pos, dim3 dim){
	return pos.x + pos.y*dim.x + pos.z*dim.x * dim.y;
}

__global__ void padImageKernel(float* fixedsrc, float* movingsrc, float* fixed, float* moving, float* deformedMoving, const dim3 dim, const dim3 padDim,  const dim3 marginSize){
	extern __shared__ float sh[];

	float* sh_fix = (float*)&sh[0];
	float* sh_moving = (float*) &sh[blockDim.x];
	
	int tidx = threadIdx.x;
	dim3 readpos = dim3(threadIdx.x, blockIdx.x, blockIdx.y);
	unsigned int readIdx = pos(readpos, dim);
	dim3 writepos = dim3(
		threadIdx.x + marginSize.x,
		blockIdx.x + marginSize.y, 
		blockIdx.y + marginSize.z
		);

	unsigned int writeIdx = pos(writepos, padDim);
	
	sh_fix[tidx] = fixedsrc[readIdx]; 
	sh_moving[tidx] = movingsrc[readIdx];
	
	__syncthreads();

	float iFixed = sh_fix[tidx];
	float iMoving = sh_moving[tidx];

#ifdef THRESHOLD
	iFixed = iFixed-THRESHOLD;
	iMoving = iMoving - THRESHOLD;

	iFixed < 0 ? iFixed = 0 : iFixed = round(iFixed*(255.0f / (255.0f - THRESHOLD)));
	iMoving < 0 ? iMoving = 0 : iMoving = round(iMoving*(255.0f / (255.0f - THRESHOLD)));
	
	sh_fix[tidx] = iFixed;
	sh_moving[tidx] = iMoving;
#endif

	fixed[writeIdx] = sh_fix[tidx];
	moving[writeIdx] = sh_moving[tidx];
	deformedMoving[writeIdx] = sh_moving[tidx];
}

__global__ void gradientKernel(float* image, float4* g, dim3 dim){
	__shared__ float sh[10][10][10];
	__shared__ float4 shg[10][10][10];

	////	//position of the thread in the image
	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
		);

	unsigned int tidx = pos(tpos, dim);
	dim3 ipos;


	// load into shared memory


	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		//first batch loading
		sh[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] = image[tidx];

		//second batch loading (TODO: Tackle thread divergence)
		if (threadIdx.x == 0 && tpos.x>0)
			sh[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] =
			image[pos(dim3(tpos.x - 1, tpos.y, tpos.z), dim)];

		if (threadIdx.y == 0 && tpos.y>0) 
			sh[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] =
			image[pos(dim3(tpos.x, tpos.y - 1, tpos.z), dim)];

		if (threadIdx.z == 0 && tpos.z>0)
			sh[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] =
			image[pos(dim3(tpos.x, tpos.y, tpos.z - 1), dim)];

		if (threadIdx.x == blockDim.x - 1 && tpos.x<(dim.x-1))
			sh[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] =
			image[pos(dim3(tpos.x + 1, tpos.y, tpos.z), dim)];

		if (threadIdx.y == blockDim.y - 1 && tpos.y<(dim.y-1))
			sh[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] =
			image[pos(dim3(tpos.x, tpos.y + 1, tpos.z), dim)];

		if (threadIdx.z == blockDim.z - 1 && tpos.z<(dim.z-1))
			sh[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] =
			image[pos(dim3(tpos.x, tpos.y, tpos.z + 1), dim)];
	}


	__syncthreads();
//
	//Gradient decomposition
	//gradient at the specific thread
	float4 tg;
	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		if (tpos.x == 0 || tpos.y == 0 || tpos.z == 0 ||
			tpos.x == dim.x - 1 || tpos.y == dim.y - 1 || tpos.z == dim.z - 1){
			tg.x = 0;
			tg.y = 0;
			tg.z = 0;
			tg.w = 0;
		}
		else
		{
			tg.x = (sh[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] -
				sh[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x]) / 2.0f;
			tg.y = (sh[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] -
				sh[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1]) / 2.0f;
			tg.z = (sh[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] -
				sh[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1]) / 2.0f;
			tg.w = tg.x*tg.x + tg.y*tg.y + tg.z*tg.z;
		}
	}

	shg[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] = tg;

	//__syncthreads();

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)

	g[tidx] = shg[ threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1];
}

__device__ int sign(float val) {
	return (float(0) < val) - (val<float(0));
}

__global__ void updateKernel( float* F,  float* Mp, float4* u, float4* uf, float alpha2, dim3 dim)
{
	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
	);
	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		float scale;
		unsigned int tidx = pos(tpos, dim);
		float diff = F[tidx] - Mp[tidx];
		float4 tu = u[tidx];
		float4 tuf = uf[tidx];

		int sgn = sign(tu.x*tuf.x + tu.y*tuf.y + tu.z*tuf.z);

		if (diff == 0 || tu.w == 0 || sgn == 0) {
			scale = 0;
		}
		else {
			scale = diff / (tu.w + alpha2*diff*diff);
			(scale < 0) ? scale *= sgn : scale;
		}

		tu.x *= scale;
		tu.y *= scale;
		tu.z *= scale;

		u[tidx] = tu;
	}
}

__global__ void gaussianKernel(float4* u, cudaTextureObject_t mask, int kernel_radius, dim3 dim, int dir)
{
	extern __shared__ float4 N_ds[];

	if (dir == 1) 
	{
		//X direction
		//block dimension:	{dim[0], 1, 1}
		//grid dimension:	{1, dim[1], dim[2]}
		//shared memory:	sizeof(float4)*(dim[0]+2*kernel_radius)

		dim3 tpos = dim3(
			threadIdx.x + blockDim.x*blockIdx.x,
			threadIdx.y + blockDim.y*blockIdx.y,
			blockIdx.z
		);
		unsigned int sh_idx = threadIdx.x + kernel_radius;
		unsigned int tidx = pos(tpos, dim);

		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			// First batch loading
			N_ds[sh_idx] = u[tidx];

			//second batch loading
			if (threadIdx.x < kernel_radius) {
				N_ds[threadIdx.x].x = 0;
				N_ds[threadIdx.x].y = 0;
				N_ds[threadIdx.x].z = 0;
				N_ds[threadIdx.x].w = 0;
			}
			else if (threadIdx.x >= dim.x - kernel_radius) {
				N_ds[sh_idx + kernel_radius].x = 0;
				N_ds[sh_idx + kernel_radius].y = 0;
				N_ds[sh_idx + kernel_radius].z = 0;
				N_ds[sh_idx + kernel_radius].w = 0;
			}
		}

		__syncthreads();

		float weight;
		float4 sum = { 0, 0, 0, 0 };
		/***** Perform Convolution *****/
		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) 
		{
			#pragma unroll
			for (int i = -kernel_radius; i <= kernel_radius; ++i) {
				weight = tex1Dfetch<float>(mask, abs(i));
				sum.x += N_ds[sh_idx + i].x * weight;
				sum.y += N_ds[sh_idx + i].y * weight;
				sum.z += N_ds[sh_idx + i].z * weight;
			}
		}
		__syncthreads();

		//save results to shared memory
		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			N_ds[sh_idx] = sum;
		}
		__syncthreads();

		//save results to global memory
		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			u[tidx] = N_ds[sh_idx];
		}
	}



	if (dir == 2)
	{
		//Y direction
		//block dimension:	{4, dim[1], 1} 

		//					(can also be {2, dim[1], 1} or {1, dim[1], 1} if # of threads exceed 1024)

		//grid dimension:	{ceil(dim[0] / 4.0), 1, dim[2]}
		//					(can be{ ceil(dim[0] / 2.0), 1, dim[2]} or{ dim[0], 1, dim[2]} 

		//shared memory:	sizeof(float4) * blockdim.x * (dim[1] + 2 * mask_radius)

		dim3 tpos = dim3(
			threadIdx.x + blockDim.x*blockIdx.x,
			threadIdx.y + blockDim.y*blockIdx.y,
			blockIdx.z
			);

		int sh_idx = threadIdx.x + (threadIdx.y + kernel_radius)*blockDim.x;
		unsigned int tidx = pos(tpos, dim);

		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			// First batch loading
			N_ds[sh_idx] = u[tidx];

			//second batch loading
			if (threadIdx.y < kernel_radius) {
				N_ds[threadIdx.x + threadIdx.y*blockDim.x].x = 0;
				N_ds[threadIdx.x + threadIdx.y*blockDim.x].y = 0;
				N_ds[threadIdx.x + threadIdx.y*blockDim.x].z = 0;
				N_ds[threadIdx.x + threadIdx.y*blockDim.x].w = 0;
			}
			else if (threadIdx.y >= dim.y - kernel_radius) {
				N_ds[sh_idx + kernel_radius*blockDim.x].x = 0;
				N_ds[sh_idx + kernel_radius*blockDim.x].y = 0;
				N_ds[sh_idx + kernel_radius*blockDim.x].z = 0;
				N_ds[sh_idx + kernel_radius*blockDim.x].w = 0;
			}
		}
		__syncthreads();

		float weight;
		float4 sum = { 0, 0, 0, 0 };

		///***** Perform Convolution *****/
		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
#pragma unroll
			for (int i = -kernel_radius; i <= kernel_radius; ++i) {
				weight = tex1Dfetch<float>(mask, abs(i));
				sum.x += N_ds[sh_idx + i*blockDim.x].x * weight;
				sum.y += N_ds[sh_idx + i*blockDim.x].y * weight;
				sum.z += N_ds[sh_idx + i*blockDim.x].z * weight;
				//sum.w = sum.w + N_ds[sh_idx + x*blockDim.x].w * weight;
			}
		}
		__syncthreads();
		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			N_ds[sh_idx] = sum;
		}
		__syncthreads();
		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			u[tidx] = N_ds[sh_idx];
		}

	}
	if (dir == 3) //Z direction
	{
		//Y direction
		//block dimension:	{4, dim[2], 1}  
		//					(can also be {2, dim[2], 1} or {1, dim[2], 1} if # of threads exceed 1024)

		//grid dimension:	{ceil(dim[0] / 4.0), dim[1], 1}
		//					(can be{ ceil(dim[0] / 2.0),  1, dim[1]} or{ dim[0],  1, dim[1]} 

		//shared memory:	sizeof(float4) * blockdim.x * (dim[2] + 2 * mask_radius)


		dim3 tpos = dim3(
			threadIdx.x + blockDim.x*blockIdx.x,
			blockIdx.y,
			threadIdx.y
			);

		int sh_idx = threadIdx.x + (threadIdx.y + kernel_radius)*blockDim.x;
		unsigned int tidx = pos(tpos, dim);

		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			// First batch loading
			N_ds[sh_idx] = u[tidx];

			//second batch loading
			if (threadIdx.y < kernel_radius) {
				N_ds[threadIdx.x + threadIdx.y*blockDim.x].x = 0;
				N_ds[threadIdx.x + threadIdx.y*blockDim.x].y = 0;
				N_ds[threadIdx.x + threadIdx.y*blockDim.x].z = 0;
				N_ds[threadIdx.x + threadIdx.y*blockDim.x].w = 0;
			}
			else if (threadIdx.y >= dim.y - kernel_radius) {
				N_ds[sh_idx + kernel_radius*blockDim.x].x = 0;
				N_ds[sh_idx + kernel_radius*blockDim.x].y = 0;
				N_ds[sh_idx + kernel_radius*blockDim.x].z = 0;
				N_ds[sh_idx + kernel_radius*blockDim.x].w = 0;
			}
		}
		__syncthreads();

		float weight;
		float4 sum = { 0,0,0,0 };
		/***** Perform Convolution *****/

		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			#pragma unroll
			for (int i = -kernel_radius; i <= kernel_radius; ++i) {
				weight = tex1Dfetch<float>(mask, abs(i));
				sum.x += N_ds[sh_idx + i*blockDim.x].x * weight;
				sum.y += N_ds[sh_idx + i*blockDim.x].y * weight;
				sum.z += N_ds[sh_idx + i*blockDim.x].z * weight;

			}
			sum.w = sum.x*sum.x + sum.y*sum.y + sum.z*sum.z;
		}
		__syncthreads();
		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			N_ds[sh_idx] = sum;
		}
		__syncthreads();
		if (tpos.x < dim.x &&
			tpos.y < dim.y &&
			tpos.z < dim.z) {
			u[tidx] = N_ds[sh_idx];
		}


	}
}

__global__ void coordinateImageKernel(float4* u, dim3 dim) {

	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
	);
	unsigned int tidx = pos(tpos, dim);

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)	{
		u[tidx].x = u[tidx].x + tpos.x;
		u[tidx].y = u[tidx].y + tpos.y;
		u[tidx].z = u[tidx].z + tpos.z;
	}
}

__global__ void composeKernel(float4* u, cudaTextureObject_t tex_v, dim3 dim){
	extern __shared__ float4 N_ds[];


	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
	);

	unsigned int tidx = pos(tpos, dim);
	unsigned int sh_idx = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		N_ds[sh_idx] = u[tidx];
		float4 tu = N_ds[sh_idx];
		float4 up = tex3D<float4>(tex_v, (float)tpos.x + tu.x+0.5f, (float)tpos.y + tu.y+0.5f, (float)tpos.z + tu.z+0.5f);
		tu.x = up.x - tpos.x;
		tu.y = up.y - tpos.y;
		tu.z = up.z - tpos.z;
		N_ds[sh_idx] = tu;
	}
	__syncthreads();

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		u[tidx] = N_ds[sh_idx];
	}
}

__global__ void selfComposeKernel(float4* s, cudaTextureObject_t tex_s, dim3 dim){
	extern __shared__ float4 N_ds[];


	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
		);

	unsigned int tidx = pos(tpos, dim);
	unsigned int sh_idx = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		N_ds[sh_idx] = s[tidx];
		float4 ts = N_ds[sh_idx];
		float4 sp = tex3D<float4>(tex_s, (float)ts.x + 0.5f, (float)ts.y + 0.5f, (float)ts.z + 0.5f);
		ts.x = sp.x - tpos.x;
		ts.y = sp.y - tpos.y;
		ts.z = sp.z - tpos.z;
		N_ds[sh_idx] = ts;
	}
	__syncthreads();

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		s[tidx] = N_ds[sh_idx];
	}
}



__global__ void  scaleKernel(float4* s, float4* v, float scale, unsigned int paddedlen){
	extern __shared__ float4 N_ds[];

	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
		);

	unsigned int tidx = tpos.x;
	unsigned int sh_idx = threadIdx.x;

	if (tidx<paddedlen)
	{
		N_ds[sh_idx] = v[tidx];
		N_ds[sh_idx].x *= scale;
		N_ds[sh_idx].y *= scale;
		N_ds[sh_idx].z *= scale;
	}
	__syncthreads();

	if (tidx<paddedlen)
	{
		s[tidx] = N_ds[sh_idx];
	}
}

__global__ void u2uw(float4* u, float* uw, unsigned int len){
	unsigned tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid<len) uw[tid] = u[tid].w;
}

__global__ void interpolateImageKernel(float* Mp, float4* s, cudaTextureObject_t tex_M, dim3 dim){

	extern __shared__ float sh[];

	float* sh_Mp = (float*)&sh[0];
	float4* sh_s = (float4*)&sh[blockDim.x*blockDim.y*blockDim.z];

	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
		);

	unsigned int tidx = pos(tpos, dim);
	unsigned int sh_idx = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		sh_s[sh_idx] = s[tidx];
		float4 ts = sh_s[sh_idx];
		sh_Mp[sh_idx] = tex3D<float>(tex_M, (float)tpos.x + ts.x+0.5f, (float)tpos.y + ts.y+0.5f, (float)tpos.z + ts.z+0.5f);
	}
	__syncthreads();

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		Mp[tidx] = sh_Mp[sh_idx];
	}
}

__global__ void energyKernel(const float* F, const float* Mp, const float4* s, float* e, float reg_weight, dim3 dim){
	__shared__ float4 sh_s[10][10][10];
	__shared__ float sh_F[8][8][8];

	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
		);

	unsigned int tidx = pos(tpos, dim);
	dim3 ipos;

	// load into shared memory

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		//first batch loading
		sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] = s[tidx];
		sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1].w = Mp[tidx];

		//second batch loading (TODO: Tackle thread divergence)
		{
			if (threadIdx.x == 0 && tpos.x>0)
				sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] =
				s[pos(dim3(tpos.x - 1, tpos.y, tpos.z), dim)];

			if (threadIdx.y == 0 && tpos.y>0)
				sh_s[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] =
				s[pos(dim3(tpos.x, tpos.y - 1, tpos.z), dim)];

			if (threadIdx.z == 0 && tpos.z>0)
				sh_s[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] =
				s[pos(dim3(tpos.x, tpos.y, tpos.z - 1), dim)];

			if (threadIdx.x == blockDim.x - 1 && tpos.x < (dim.x - 1))
				sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] =
				s[pos(dim3(tpos.x + 1, tpos.y, tpos.z), dim)];

			if (threadIdx.y == blockDim.y - 1 && tpos.y < (dim.y - 1))
				sh_s[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] =
				s[pos(dim3(tpos.x, tpos.y + 1, tpos.z), dim)];

			if (threadIdx.z == blockDim.z - 1 && tpos.z < (dim.z - 1))
				sh_s[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] =
				s[pos(dim3(tpos.x, tpos.y, tpos.z + 1), dim)];
		}

		sh_F[threadIdx.z][threadIdx.y][threadIdx.x] = F[tidx];
	}

	__syncthreads();


	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
	{
		//compute difference squared and save to sh_F
		sh_F[threadIdx.z][threadIdx.y][threadIdx.x] -= sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1].w;

		sh_F[threadIdx.z][threadIdx.y][threadIdx.x] =
			sh_F[threadIdx.z][threadIdx.y][threadIdx.x] *
			sh_F[threadIdx.z][threadIdx.y][threadIdx.x] /
			(dim.x*dim.y*dim.z);

		//compute (jacobian+identity)
		float3 gx, gy, gz;

		if (tpos.x == 0 || tpos.y == 0 || tpos.z == 0 ||
			tpos.x == dim.x - 1 || tpos.y == dim.y - 1 || tpos.z == dim.z - 1){
			//At margin, Jac = 0;
			gx = { 1, 0, 0 };
			gy = { 0, 1, 0 };
			gz = { 0, 0, 1 };
		}
		else
		{
			gx.x = ((sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2].x -
				sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x].x) / 2.0f) + 1.0f;
			gx.y = (sh_s[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1].x -
				sh_s[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1].x) / 2.0f;
			gx.z = (sh_s[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1].x -
				sh_s[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1].x) / 2.0f;

			gy.x = (sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2].y -
				sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x].y) / 2.0f;
			gy.y = ((sh_s[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1].y -
				sh_s[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1].y) / 2.0f) + 1.0f;
			gy.z = (sh_s[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1].y -
				sh_s[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1].y) / 2.0f;

			gz.x = (sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2].z -
				sh_s[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x].z) / 2.0f;
			gz.y = (sh_s[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1].z -
				sh_s[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1].z) / 2.0f;
			gz.z = ((sh_s[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1].z -
				sh_s[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1].z) / 2.0f) + 1.0f;
		}

		float jac =
			gx.x * gy.y * gz.z +
			gy.x * gz.y * gx.z +
			gz.x * gx.y * gy.z -
			gz.x * gy.y * gx.z -
			gy.x * gx.y * gz.z -
			gx.x * gz.y * gy.z;

		jac = jac*jac*reg_weight / (dim.x*dim.y*dim.z);

		sh_F[threadIdx.z][threadIdx.y][threadIdx.x] += jac;
	}

	__syncthreads();

	if (tpos.x < dim.x &&
		tpos.y < dim.y &&
		tpos.z < dim.z)
		e[tidx] = sh_F[threadIdx.z][threadIdx.y][threadIdx.x];
}