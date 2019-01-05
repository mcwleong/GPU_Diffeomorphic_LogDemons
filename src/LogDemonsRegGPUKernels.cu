#include "LogDemonsRegGPUKernels.cuh"

__global__ void gradientKernel(float* __restrict__ I, float* fg, int dir, dim3 dim){
	extern  __shared__ float sh[];

	dim3 tpos;
	unsigned int tidx;
	unsigned int sh_idx;

	float tg;


		// x direction
	if (dir == 0){
		// blocksize = {dim[0], 1, 1}
		// gridsize = {dim[1], dim[2], 1)
		// shared memory size = dim[0] + 2

		tpos = dim3(
			threadIdx.x,
			blockIdx.x,
			blockIdx.y
		);
		tidx = gpos(tpos, dim);

		sh_idx = threadIdx.x + 1;

		// Initialize shared emmory
		if (tpos < dim){
			sh[sh_idx] = I[tidx];
			if (threadIdx.x == 0){
				sh[sh_idx - 1] = 0.0f;
			} else if (threadIdx.x == blockDim.x - 1){
				sh[sh_idx + 1] = 0.0f;
			}
		}

		__syncthreads();

		// Gradient decomposition
		if (tpos < dim)
		{
			// tg = __fdividef(sh[sh_idx + 1] - sh[sh_idx - 1], 2.0f); //Use intrinsic and --use_fast_math compiler options to yield higher performance
			tg = (sh[sh_idx + 1] - sh[sh_idx - 1] )/ 2.0f;
			fg[tidx] = tg;
		}
	}
	// y direction
	else if (dir == 1){
		//blocksize = {32, 32, 1}
		//gridsize = {ceil(dim[0])/32, dim[2], 1}
		//shmemsize = 32*(dim[1]+2)*sizeof(float)


		//fetch data into shared memory
		//Instruction-level parallelism

		unsigned int shsize = (dim.y + 2) * 32;
		for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < shsize; i += blockDim.x*blockDim.y){
			sh[i] = 0;
		}
		__syncthreads();

		for (int j = threadIdx.y; j < dim.y; j += blockDim.y) {
			tpos = dim3(
				threadIdx.x + blockIdx.x*blockDim.x,
				j,
				blockIdx.y);
			tidx = gpos(tpos, dim);
			sh_idx = threadIdx.x + (j+1) * blockDim.x;
			if (tpos < dim){
				sh[sh_idx] = I[tidx];
			}
			
		}
		__syncthreads();

	
		// Gradient decomposition
		for (int j = threadIdx.y; j < dim.y; j += blockDim.y) {
			//compute the position and corresponding index in shared memory
			tpos = dim3(
				threadIdx.x + blockIdx.x*blockDim.x,
				j,
				blockIdx.y);
			sh_idx = threadIdx.x + (j+1)*blockDim.x;
			tidx = gpos(tpos, dim);

			if (tpos < dim){ //32-pencil block might overflow the x-dimension
				//tg = __fdividef((sh[sh_idx + blockDim.x] - sh[sh_idx - blockDim.x]), 2.0f);
				tg = (sh[sh_idx + blockDim.x] - sh[sh_idx - blockDim.x]) / 2.0f;
				fg[tidx] = tg;
			}
		}
	}
	//Z direction
	else {

		//pass Z
		//blocksize = {32, 32, 1}
		//gridsize = {ceil(dim[0])/32, dim[1], 1}
		//memsize = 32*(dim[2]+2)*sizeof(float)

		dim3 tpos;


		unsigned int shsize = (dim.z + 2) * 32;
		for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < shsize; i += blockDim.x*blockDim.y){
			sh[i] = 0;
		}
		__syncthreads();

		for (int k = threadIdx.y; k < dim.z; k += blockDim.y) {
			tpos = dim3(
				threadIdx.x + blockIdx.x*blockDim.x,
				blockIdx.y,
				k);
			tidx = gpos(tpos, dim);
			sh_idx = threadIdx.x + (k+1) * blockDim.x;
			if (tpos < dim){
				sh[sh_idx] = I[tidx];
			}
		}
		__syncthreads();

		// Gradient decomposition
		for (int  k = threadIdx.y; k < dim.z; k += blockDim.y) {
			//compute the position and corresponding index in shared memory
			tpos = dim3(
				threadIdx.x + blockIdx.x*blockDim.x,
				blockIdx.y,
				k);
			sh_idx = threadIdx.x + (k+1)*blockDim.x;
			tidx = gpos(tpos, dim);

			if (tpos < dim){ //32-pencil block might overflow the x-dimension
				tg = (sh[sh_idx + blockDim.x] - sh[sh_idx - blockDim.x]) / 2.0f;
				fg[tidx] = tg;
			}
		}
	}
}

/* Requires further optimization */
__global__ void updateKernel(float* F, float* Mp, float* ux, float* uy, float* uz, float* uxf, float* uyf, float* uzf,  float alpha2, unsigned int len){

	unsigned int tidx = blockIdx.x* blockDim.x + threadIdx.x;
	if (tidx <len)
	{
		float diff = F[tidx] - Mp[tidx];
		float tux = ux[tidx];
		float tuy = uy[tidx];
		float tuz = uz[tidx];
		float tu2 = tux*tux + tuy*tuy + tuz*tuz;

		float scale = diff / (tu2 + alpha2*diff*diff);
		float sign =  sgn(tux*uxf[tidx] + tuy*uyf[tidx] + tuz*uzf[tidx]);

		if (diff == 0 || tu2 == 0) {
			scale = 0;
		}
		if (scale<0) scale *= sign;


		tux *= scale;
		tuy *= scale;
		tuz *= scale;

		ux[tidx] = tux;
		uy[tidx] = tuy;
		uz[tidx] = tuz;
	}
}

__global__ void gaussianKernel(float* I, cudaTextureObject_t tex, dim3 dim, int radius, int pass){
	extern __shared__ float sh[];

	unsigned int tidx;
	float sum = 0;
	int window_size = radius +1;
	//pass X
	//blocksize = {dim[0], 1, 1}
	//gridsize = {dim[1], dim[2], 1}

	if (pass == 0){
		dim3 tpos = dim3(
			threadIdx.x,
			blockIdx.x,
			blockIdx.y);

		tidx = gpos(tpos,dim);

		// Initialize shared memory
		int shsize = (blockDim.x + radius * 2);
		for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < shsize; i += blockDim.x*blockDim.y){
			sh[i] = 0;
		}
		__syncthreads();

		//fetch data into shared memory
		int sh_idx = threadIdx.x + radius;
		sh[sh_idx] = I[tidx];


		__syncthreads();

		//Convolution
		if (tpos<dim){
			for (int i = -window_size + 1; i < window_size; ++i){
				sum += tex1Dfetch<float>(tex, abs(i))*sh[sh_idx + i];
			}
		}
		I[tidx] =  sum;
	}

	//pass Y
	//blocksize = {32, 32, 1}
	//gridsize = {ceil(dim[0])/32, dim[2], 1}
	//memsize = 32*dim[1]*sizeof(float)
	else if (pass == 1)
	{
		dim3 tpos;

		unsigned int sh_idx;
		int shsize = 32 * (dim.y+radius*2);

		// Initialize shared memory
		for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < shsize; i += blockDim.x*blockDim.y){
			sh[i] = 0;
		}
		__syncthreads();
		//fetch data into shared memory using block-based ILP
		for (int j = threadIdx.y; j < dim.y; j += blockDim.y) {
			tpos = dim3(
				threadIdx.x + blockIdx.x*blockDim.x,
				j,
				blockIdx.y);
			tidx = gpos(tpos, dim);
			sh_idx = threadIdx.x + (j+radius) * blockDim.x;
			if (tpos < dim){
				sh[sh_idx] = I[tidx];
			}
		}
		__syncthreads();

		//Convolution
		for (int j = threadIdx.y; j < dim.y; j += blockDim.y) {
			//compute the position and corresponding index in shared memory
			sum = 0;
			tpos = dim3(
				threadIdx.x + blockIdx.x*blockDim.x,
				j,
				blockIdx.y);
			sh_idx = threadIdx.x + (j + radius) * blockDim.x;
			tidx = gpos(tpos, dim);

			if (tpos < dim){ // To avoid out-of-bound access along the x dimension
				for (int i = -window_size + 1; i < window_size; ++i){
					sum += tex1Dfetch<float>(tex, abs(i))*sh[sh_idx + blockDim.x * i];
				}
				I[tidx] = sum;
			}
		}
	}

	//pass Z
	//blocksize = {32, 32, 1}
	//gridsize = {ceil(dim[0])/32, dim[1], 1}
	//memsize = 32*dim[2]*sizeof(float)
	else
	{
		dim3 tpos;

		unsigned int sh_idx;
		int shsize = 32 * (dim.z + radius * 2);

		// Initialize shared memory
		for (int i = threadIdx.x + threadIdx.y*blockDim.x; i < shsize; i += blockDim.x*blockDim.y){
			sh[i] = 0;
		}
		__syncthreads();


		//fetch data into shared memory
		for (int k = threadIdx.y; k < dim.z; k += blockDim.y) {
			tpos = dim3(
				threadIdx.x + blockIdx.x*blockDim.x,
				blockIdx.y,
				k);
			tidx = gpos(tpos, dim);
			sh_idx = threadIdx.x + (k+radius) * blockDim.x;
			if (tpos < dim){
				sh[sh_idx] = I[tidx];
			}
		}
		__syncthreads();

		//Convolution
		for (int k = threadIdx.y; k < dim.z; k += blockDim.y) {
			sum = 0;
			//compute the position and corresponding index in shared memory
			tpos = dim3(
				threadIdx.x + blockIdx.x*blockDim.x,
				blockIdx.y,
				k);
			sh_idx = threadIdx.x + (k + radius) * blockDim.x;
			tidx = gpos(tpos, dim);

			if (tpos < dim){ //32-pencil block might overflow the x-dimension
				for (int i = -window_size + 1; i < window_size; ++i){
					sum += tex1Dfetch<float>(tex, abs(i))*sh[sh_idx + blockDim.x * i];
				}
				I[tidx] = sum;
			}
		}
	}
}



__global__ void coordinateImageKernel(float *vx, float* vy, float* vz, dim3 dim, AddSubtractCoordinateImage op){
	// blocksize: dim[0] 1 1
	// gridsize: dim[1] dim[2] 1
	
	dim3 tpos = dim3(
		threadIdx.x,
		blockIdx.x,
		blockIdx.y
		);

	unsigned int tidx = gpos(tpos, dim);

	if (tpos < dim){ 
		if (op == addCoordinate){
			vx[tidx] = vx[tidx] + threadIdx.x;
			vy[tidx] = vy[tidx] + blockIdx.x;
			vz[tidx] = vz[tidx] + blockIdx.y;
		}
		else
		{
			vx[tidx] = vx[tidx] - threadIdx.x;
			vy[tidx] = vy[tidx] - blockIdx.x;
			vz[tidx] = vz[tidx] - blockIdx.y;
		}
	}
}

__global__ void interpolateImageKernel(cudaTextureObject_t tex, float* sx, float* sy, float* sz, float* Ip, dim3 dim){
	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
		);
	unsigned int tidx = gpos(tpos, dim);

	if (tpos < dim){
		//A much faster way but it is way less precise
		Ip[tidx] = tex3D<float>(tex, sx[tidx] + 0.5f, sy[tidx] + 0.5f, sz[tidx] + 0.5f);


	/*	float ind_x = floor(sx[tidx]) + 0.5f;
		float xd =  sx[tidx] - ind_x + 0.5f;
		float ind_y = floor(sy[tidx]) + 0.5f;
		float yd =  sy[tidx] - ind_y + 0.5f;
		float ind_z = floor(sz[tidx]) + 0.5f;
		float zd = sz[tidx] - ind_z + 0.5f;

		const float d000 = tex3D<float>(tex, ind_x, ind_y, ind_z);
		const float d001 = tex3D<float>(tex, ind_x + 1, ind_y, ind_z);
		const float d010 = tex3D<float>(tex, ind_x, ind_y + 1, ind_z);
		const float d011 = tex3D<float>(tex, ind_x + 1, ind_y + 1, ind_z);
		const float d100 = tex3D<float>(tex, ind_x, ind_y, ind_z + 1);
		const float d101 = tex3D<float>(tex, ind_x + 1, ind_y, ind_z + 1);
		const float d110 = tex3D<float>(tex, ind_x, ind_y + 1, ind_z + 1);
		const float d111 = tex3D<float>(tex, ind_x + 1, ind_y + 1, ind_z + 1);

		float d00 = xd*d001 + (1 - xd)*d000;
		float d01 = xd*d011 + (1 - xd)*d010;
		float d10 = xd*d101 + (1 - xd)*d100;
		float d11 = xd*d111 + (1 - xd)*d110;

		float d0 = yd * d01 + (1 - yd) * d00;
		float d1 = yd * d11 + (1 - yd) * d10;

		Ip[tidx] = zd * d1 + (1 - zd) * d0;*/
	
	}
}

__global__ void normalizeVectorKernel(float* fx, float* fy, float* fz, float* norm2, unsigned int len){
	unsigned int tidx = blockIdx.x* blockDim.x + threadIdx.x;
	if (tidx < len){
		norm2[tidx] = fx[tidx] * fx[tidx] + fy[tidx] * fy[tidx] + fz[tidx] * fz[tidx];
	}
}

__global__ void scaleKernel(float* vx, float* vy, float* vz, float* sx, float* sy, float* sz, float scale, unsigned int len){
	unsigned int tidx = blockIdx.x* blockDim.x + threadIdx.x;
	if (tidx < len){
		sx[tidx] = vx[tidx] * scale;
		sy[tidx] = vy[tidx] * scale;
		sz[tidx] = vz[tidx] * scale;
	}
}



__global__ void jacobianKernel(float* sx, float* sy, float*sz, float* jac2, dim3 dim){
	
	__shared__ float sh_sx[10][10][10];
	__shared__ float sh_sy[10][10][10];
	__shared__ float sh_sz[10][10][10];


	dim3 tpos = dim3(
		threadIdx.x + blockIdx.x*blockDim.x,
		threadIdx.y + blockIdx.y*blockDim.y,
		threadIdx.z + blockIdx.z*blockDim.z
		);

	unsigned int tidx = gpos(tpos, dim);
	//dim3 ipos;

	// load into shared memory

	if (tpos < dim)
	{
		//first batch loading
		sh_sx[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] = sx[tidx];
		sh_sy[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] = sy[tidx];
		sh_sz[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] = sz[tidx];


		//second batch loading: Handle boundary conditions
		if (threadIdx.x == 0){
			if (tpos.x == 0){
				sh_sx[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] = 0;
				sh_sy[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] = 0;
				sh_sz[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] = 0;

			}
			else {
				sh_sx[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] = sx[gpos(dim3(tpos.x - 1, tpos.y, tpos.z), dim)];
				sh_sy[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] = sy[gpos(dim3(tpos.x - 1, tpos.y, tpos.z), dim)];
				sh_sz[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] = sz[gpos(dim3(tpos.x - 1, tpos.y, tpos.z), dim)];
			}
		}

		if (threadIdx.y == 0){
			if (tpos.y == 0){
				sh_sx[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] = 0;
				sh_sy[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] = 0;
				sh_sz[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] = 0;
			}
			else {
				sh_sx[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] = sx[gpos(dim3(tpos.x, tpos.y - 1, tpos.z), dim)];
				sh_sy[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] = sy[gpos(dim3(tpos.x, tpos.y - 1, tpos.z), dim)];
				sh_sz[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] = sz[gpos(dim3(tpos.x, tpos.y - 1, tpos.z), dim)];
			}
		}


		if (threadIdx.z == 0){
			if (tpos.z == 0){
				sh_sx[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] = 0;
				sh_sy[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] = 0;
				sh_sz[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] = 0;
			}
			else {
				sh_sx[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] = sx[gpos(dim3(tpos.x, tpos.y, tpos.z - 1), dim)];
				sh_sy[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] = sy[gpos(dim3(tpos.x, tpos.y, tpos.z - 1), dim)];
				sh_sz[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] = sz[gpos(dim3(tpos.x, tpos.y, tpos.z - 1), dim)];
			}
		}


		if (threadIdx.x == 7){
			if (tpos.x == dim.x - 1){
				sh_sx[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] = 0;
				sh_sy[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] = 0;
				sh_sz[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] = 0;
			}
			else {
				sh_sx[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] = sx[gpos(dim3(tpos.x + 1, tpos.y, tpos.z), dim)];
				sh_sy[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] = sy[gpos(dim3(tpos.x + 1, tpos.y, tpos.z), dim)];
				sh_sz[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] = sz[gpos(dim3(tpos.x + 1, tpos.y, tpos.z), dim)];
			}
		}

		if (threadIdx.y == 7){
			if (tpos.y == dim.y - 1){
				sh_sx[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] = 0;
				sh_sy[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] = 0;
				sh_sz[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] = 0;
			}
			else {
				sh_sx[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] = sx[gpos(dim3(tpos.x, tpos.y + 1, tpos.z), dim)];
				sh_sy[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] = sy[gpos(dim3(tpos.x, tpos.y + 1, tpos.z), dim)];
				sh_sz[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] = sz[gpos(dim3(tpos.x, tpos.y + 1, tpos.z), dim)];
			}
		}

		if (threadIdx.z == 7){
			if (tpos.z == dim.z - 1){
				sh_sx[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] = 0;
				sh_sy[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] = 0;
				sh_sz[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] = 0;
			}
			else {
				sh_sx[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] = sx[gpos(dim3(tpos.x, tpos.y, tpos.z + 1), dim)];
				sh_sy[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] = sy[gpos(dim3(tpos.x, tpos.y, tpos.z + 1), dim)];
				sh_sz[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] = sz[gpos(dim3(tpos.x, tpos.y, tpos.z + 1), dim)];
			}
		}
	}

	__syncthreads();


	if (tpos<dim)
	{
		//compute (jacobian+identity)
		float3 gx, gy, gz;

		if (tpos.x == 0 || tpos.y == 0 || tpos.z == 0 ||
			tpos.x == dim.x - 1 || tpos.y == dim.y - 1 || tpos.z == dim.z - 1){
			//At margin, Jac = 0;
			gx = { 1.0f, 0.0f, 0.0f };
			gy = { 0.0f, 1.0f, 0.0f };
			gz = { 0.0f, 0.0f, 1.0f };
		}
		else
		{
			gx.x = ((sh_sx[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] -
				sh_sx[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x]) / 2.0f) + 1.0f;
			gx.y = (sh_sx[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] -
				sh_sx[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1]) / 2.0f;
			gx.z = (sh_sx[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] -
				sh_sx[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1]) / 2.0f;

			gy.x = (sh_sy[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] -
				sh_sy[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x]) / 2.0f;
			gy.y = ((sh_sy[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] -
				sh_sy[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1]) / 2.0f) + 1.0f;
			gy.z = (sh_sy[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] -
				sh_sy[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1]) / 2.0f;

			gz.x = (sh_sz[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 2] -
				sh_sz[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x]) / 2.0f;
			gz.y = (sh_sz[threadIdx.z + 1][threadIdx.y + 2][threadIdx.x + 1] -
				sh_sz[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1]) / 2.0f;
			gz.z = ((sh_sz[threadIdx.z + 2][threadIdx.y + 1][threadIdx.x + 1] -
				sh_sz[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1]) / 2.0f) + 1.0f;
		}

		float jac =
			gx.x * gy.y * gz.z +
			gy.x * gz.y * gx.z +
			gz.x * gx.y * gy.z -
			gz.x * gy.y * gx.z -
			gy.x * gx.y * gz.z -
			gx.x * gz.y * gy.z;

		jac = jac*jac;

		jac2[tidx] += jac;
	}
}

__global__ void energyKernel(float* Mp, float* F, float* jac2, float* en, float reg_weight, unsigned int len){
	unsigned int tidx = threadIdx.x + blockIdx.x*blockDim.x;
	float diff2 = Mp[tidx] - F[tidx];
	diff2 = diff2*diff2;
	
	en[tidx] = diff2;//+reg_weight*jac2[tidx];
}