#include "RegistrationGPU.cuh"

RegistrationGPU::RegistrationGPU(float* Fixed, float* Moving, unsigned int dimension[3])
	:Registration(Fixed, Moving, dimension)
{
	cudaDeviceReset();
}

RegistrationGPU::~RegistrationGPU()
{
	if (GPUinitialized)
	{
		cudaFree(d_fixed);
		cudaFree(d_moving);
		cudaFree(d_deformedMoving);
		cudaFree(d_u);
		cudaFree(d_s);
		cudaFree(d_maskf);
		cudaFree(d_maskd);

		cudaDestroyTextureObject(d_texf);
		cudaDestroyTextureObject(d_texd);
	}
	cudaDeviceReset();
}

void RegistrationGPU::printDevProp(int deviceID){
	printf("CUDA Device Query...\n");
	printf("CUDA Device #%d\n", deviceID);

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, deviceID);

	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Name:                          %s\n", devProp.name);
	printf("Total global memory:           %u\n", devProp.totalGlobalMem);
	printf("Total shared memory per block: %u\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n", devProp.clockRate);
	printf("Total constant memory:         %u\n", devProp.totalConstMem);
	printf("Texture alignment:             %u\n", devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
	return;
}

void RegistrationGPU::getCudaError(const char* s){
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess){
		cerr << s << " failed!" << endl;
		cerr << cudaGetErrorString(cudaStatus) << endl;
		getchar();
		exit(1);
	}
	else
	{
		if (debug)
		cout << s << " success" << endl;
	}
}


void RegistrationGPU::SetDeviceID(int i){
	this->deviceID = i;
}

void RegistrationGPU::initialize(){
	Registration::initialize();

	e_cpu = new float[opt.iteration_max];

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(deviceID);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		getchar();
		exit(1);
	}
	cout << "GPU initialization success!" << endl;

	printDevProp(deviceID);

	padDim = dim3(paddedDim[0], paddedDim[1], paddedDim[2]);
		
	mask_width_f = int(rint(opt.sigma_f * 3));
	mask_width_d = int(rint(opt.sigma_d * 3));

	//Calculate gaussian mask and bind to texture memory
	gaussianMask_f = new float[mask_width_f + 1];
	gaussianMask_d = new float[mask_width_d + 1];

#pragma unroll
	for (int i = 0; i <= mask_width_f; ++i){
		gaussianMask_f[i] = (0.39894228f / opt.sigma_f) * exp(-(i*i) / (2 * opt.sigma_f*opt.sigma_f));
		printf("%f\n", gaussianMask_f[i]);
	}

#pragma unroll
	for (int i = 0; i <= mask_width_d; ++i)
		gaussianMask_d[i] = 0.39894228f / opt.sigma_d * exp(-(i*i) / (2 * opt.sigma_d*opt.sigma_d));

	cudaMalloc(&d_maskf, sizeof(float)*(mask_width_f + 1));
	getCudaError("cudaMalloc d_maskf");
	cudaMemcpy(d_maskf, gaussianMask_f, sizeof(float)*(mask_width_f + 1), cudaMemcpyHostToDevice);
	getCudaError("cudaMemcpy d_maskf");

	//set texture resources
	memset(&Res_maskf, 0, sizeof(Res_maskf));
	Res_maskf.resType = cudaResourceTypeLinear;
	Res_maskf.res.linear.devPtr = d_maskf;
	Res_maskf.res.linear.desc.f = cudaChannelFormatKindFloat;
	Res_maskf.res.linear.desc.x = 32;
	Res_maskf.res.linear.sizeInBytes = (mask_width_f + 1)*sizeof(float);

	memset(&Tex_maskf, 0, sizeof(Tex_maskf));
	Tex_maskf.readMode = cudaReadModeElementType;

	d_texf = 0;
	cudaCreateTextureObject(&d_texf, &Res_maskf, &Tex_maskf, NULL);



	cudaMalloc(&d_maskd, sizeof(float)*(mask_width_d + 1));
	getCudaError("cudaMalloc d_maskd");
	cudaMemcpy(d_maskd, gaussianMask_d, sizeof(float)*(mask_width_d + 1), cudaMemcpyHostToDevice);
	getCudaError("cudaMemcpy d_maskd");

	//set texture resources
	memset(&Res_maskd, 0, sizeof(Res_maskd));
	Res_maskd.resType = cudaResourceTypeLinear;
	Res_maskd.res.linear.devPtr = d_maskd;
	Res_maskd.res.linear.desc.f = cudaChannelFormatKindFloat;
	Res_maskd.res.linear.desc.x = 32;
	Res_maskd.res.linear.sizeInBytes = (mask_width_d + 1)*sizeof(float);

	memset(&Tex_maskd, 0, sizeof(Tex_maskd));
	Tex_maskd.readMode = cudaReadModeElementType;

	d_texd = 0;
	cudaCreateTextureObject(&d_texd, &Res_maskd, &Tex_maskd, NULL);


	//pad images
	this->padImage();
	cout << "Image Padded" << endl;

	//Initialize u, v, s and e and memset to 0
	cudaMalloc(&d_u, sizeof(float4)*paddedlen);
	getCudaError("cudaMalloc d_u");
	cudaMemset(d_u, 0, sizeof(float4)*paddedlen);
	getCudaError("cudaMemset d_u");
	cudaMalloc(&d_uf, sizeof(float4)*paddedlen);
	getCudaError("cudaMemset d_uf");
	cudaMemset(d_uf, 0, sizeof(float4)*paddedlen);
	getCudaError("cudaMemset d_uf");
	cudaMalloc(&d_v, sizeof(float4)*paddedlen);
	getCudaError("cudaMalloc d_v");
	cudaMemset(d_v, 0, sizeof(float4)*paddedlen);
	getCudaError("cudaMemset d_v");
	cudaMalloc(&d_s, sizeof(float4)*paddedlen);
	getCudaError("cudaMalloc d_s");
	cudaMemset(d_s, 0, sizeof(float4)*paddedlen);
	getCudaError("cudaMemset d_s");
	cudaMalloc(&d_e, sizeof(float)*paddedlen);
	getCudaError("cudaMalloc d_e");
	cudaMemset(d_e, 0, sizeof(float)*paddedlen);
	getCudaError("cudaMemset d_e");
	cudaMalloc(&d_uw, sizeof(float)*paddedlen);
	getCudaError("cudaMalloc d_uw");
	cudaMemset(d_uw, 0, sizeof(float)*paddedlen);
	getCudaError("cudaMemset d_uw");

	cout <<"paddedlen: "<< paddedlen << endl;
	cout << "size: " << sizeof(float4)*paddedlen << endl;

	cudaChannelFormatDesc channelDescfloat = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(
		&d_cuArr_float,
		&channelDescfloat,
		make_cudaExtent(
		paddedDim[0],
		paddedDim[1],
		paddedDim[2]
		),
		0);

	cudaChannelFormatDesc channelDescfloat4 = cudaCreateChannelDesc<float4>();
	cudaMalloc3DArray(
		&d_cuArr_float4,
		&channelDescfloat4,
		make_cudaExtent(
		paddedDim[0],
		paddedDim[1],
		paddedDim[2]
		),
		0);
	//Create tex object for wraping the moving image
	d_tex_M = CreateTextureObject<float>(d_moving);

	GPUinitialized = true;
}

void RegistrationGPU::padImage(){

	size_t srclen = sizeof(float)*dim[0] * dim[1] * dim[2];
	size_t padlen = sizeof(float)*paddedlen;

	//allocate GPU memory for the source image;
	float* d_fixedsrc, *d_movingsrc;
	cudaMalloc(&d_fixedsrc, srclen);
	getCudaError("cudaMalloc d_fixedsrc");
	cudaMalloc(&d_movingsrc, srclen);
	getCudaError("cudaMalloc d_movingsrc");

	//fetch unpadded image into gpu
	cudaMemcpy(d_fixedsrc, fixedsrc->data, srclen, cudaMemcpyHostToDevice);
	getCudaError("cudaMemcpy d_fixedsrc");
	cudaMemcpy(d_movingsrc, movingsrc->data, srclen, cudaMemcpyHostToDevice);
	getCudaError("cudaMemcpy d_movingsrc");

	// allocate memory for padded images
	cudaMalloc(&d_fixed, padlen);
	getCudaError("cudaMalloc d_fixed");
	cudaMemset(d_fixed, 0, padlen);
	getCudaError("cudaMemset d_fixed");

	cudaMalloc(&d_moving, padlen);
	getCudaError("cudaMalloc d_moving");
	cudaMemset(d_moving, 0, padlen);
	getCudaError("cudaMemset d_moving");

	cudaMalloc(&d_deformedMoving, padlen);
	getCudaError("cudaMalloc d_deformedMoving");
	cudaMemset(d_deformedMoving, 0, padlen);
	getCudaError("cudaMemset d_deformedMoving");

	dim3 blocksize = dim3(dim[0], 1, 1);
	dim3 gridsize = dim3(dim[1], dim[2], 1);

	if (debug){
		cout << "Padding image..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	//launch kernel to copy and pad the images.
	padImageKernel KERNEL_ARGS3(gridsize, blocksize, 2 * blocksize.x *sizeof(float)) (d_fixedsrc, d_movingsrc, d_fixed, d_moving, d_deformedMoving, dim3(dim[0], dim[1], dim[2]), dim3(paddedDim[0], paddedDim[1], paddedDim[2]), dim3(marginSize[0], marginSize[1], marginSize[2]));

	getCudaError("PadImageKernel");

	//release the memory allocated for the source images.
	cudaFree(d_fixedsrc);
	cudaFree(d_movingsrc);
}

void RegistrationGPU::Register(){
	
	high_resolution_clock::time_point t1;
	high_resolution_clock::time_point t2;

	char* c_it = new char;
	int run_iter = 0;
	stringstream filename;

	initialize();
	
	t1 = high_resolution_clock::now();

	for (int iter = 0; iter < opt.iteration_max; ++iter){
		++run_iter;

		filename.str("");
		filename << "moving_gpu_" << itoa(iter, c_it, 10) << ".bin";

		findupdate(d_fixed, d_deformedMoving, d_u);

		imgaussian(d_u, d_texf, mask_width_f);
		compose_count++;
		compose(d_v, d_u);

		imgaussian(d_v, d_texd, mask_width_d);

		expfield(d_v, d_s);
		iminterpolate(d_deformedMoving, d_s);

		e[iter] = energy(d_fixed, d_deformedMoving, d_s);
		printf("Iteration %i - Energy: %f (GPU)\n", iter + 1, e[iter]);
		if (iter > 4){
			if ((e[iter - 5] - e[iter]) < (e[0] * opt.stop_criterium)){
				break;
			}
		}

	}
	t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1);
	printf("\nRegistration Complete\n");
	cout << "time required for " << run_iter << " iterations: " << duration.count() << " ms" << endl;
	printf("\n\n"); 
	for (int iter = 0; iter < run_iter; ++iter){
		printf("Iteration %i - Energy: %f (GPU)\n", iter + 1, e[iter]);
	}

	printf("\nCompose count: %i\n", compose_count);

}

void RegistrationGPU::saveResult(float* d_Mp, float4* d_s){
	Savetobin(d_deformedMoving, "Mp_gpu.bin");
	Savetobin(d_s, 0, "sx_gpu.bin");
	Savetobin(d_s, 1, "sy_gpu.bin");
	Savetobin(d_s, 2, "sz_gpu.bin");
}


void RegistrationGPU::findupdate(float* d_F, float* d_Mp, float4* d_u){


	gradient(d_F, d_uf);
	gradient(d_Mp, d_u);


	float alpha2 = (opt.sigma_i*opt.sigma_i) / (opt.sigma_x*opt.sigma_x);
	
	dim3 blocksize = dim3(8, 8, 8);
	dim3 gridsize = dim3(ceil(double(paddedDim[0]) / double(blocksize.x)), ceil(double(paddedDim[1]) / double(blocksize.y)), ceil(double(paddedDim[2]) / double(blocksize.z)));

	if (debug){
		cout << "Computing update kernel..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}
	updateKernel KERNEL_ARGS2(gridsize, blocksize) (d_F, d_Mp, d_u, d_uf, alpha2, padDim);

	getCudaError("updateKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize updateKernel");
}

void RegistrationGPU::gradient(float* d_I, float4* d_g){
	dim3 blocksize = dim3(8, 8, 8);
	dim3 gridsize = dim3(ceil(double(paddedDim[0]) / double(blocksize.x)), ceil(double(paddedDim[1]) / double(blocksize.y)), ceil(double(paddedDim[2]) / double(blocksize.z)));

	if (debug){
		cout << "Gradient decomposing image..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	gradientKernel KERNEL_ARGS2(gridsize, blocksize) (d_I, d_g, padDim);
	getCudaError("gradientKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize gradientKernel");
}

void RegistrationGPU::imgaussian(float4* d_u, cudaTextureObject_t mask, int mask_size){
	//X-direction
	dim3 blocksize(paddedDim[0], 1, 1);
	dim3 gridsize(1, paddedDim[1], paddedDim[2]);
	int shsize = sizeof(float4)*(blocksize.x + mask_size * 2);

	if (debug){
		cout << "Gaussian blur... X direction" << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_u, mask, mask_size, padDim, 1);
	getCudaError("gaussianKernel - X");
	cudaDeviceSynchronize();

	//Y-direciton
	blocksize = { 4, paddedDim[1], 1 };
	gridsize = { unsigned int(ceil(paddedDim[0] / 4.0f)), 1, paddedDim[2] };

	while (blocksize.x*blocksize.y > 1024)
	{
		if (blocksize.x == 1){
			cerr << "Image must be smaller than 1024!" << endl;
			exit(1);
		}
		blocksize.x /= 2;
		gridsize.x *= 2;
	}

	shsize = sizeof(float4) * blocksize.x * (blocksize.y + 2 * mask_size);

	if (debug){
		cout << "Gaussian blur... Y direction" << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_u, mask, mask_size, padDim, 2);
	getCudaError("gaussianKernel - Y");
	cudaDeviceSynchronize();

	//Z-direciton
	blocksize = { 4, paddedDim[2], 1 };
	gridsize = { unsigned int(ceil(paddedDim[0] / 4.0f)), paddedDim[1], 1 };

	while (blocksize.x*blocksize.y > 1024)
	{
		if (blocksize.x == 1){
			cerr << "Image must be smaller than 1024!" << endl;
			exit(1);
		}
		blocksize.x /= 2;
		gridsize.x *= 2;
	}

	shsize = sizeof(float4) * blocksize.x * (blocksize.y + 2 * mask_size);

	if (debug){
		cout << "Gaussian blur... Z direction" << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_u, mask, mask_size, padDim, 3);
	getCudaError("gaussianKernel - Z");
	cudaDeviceSynchronize();
}

template <typename T>
cudaTextureObject_t RegistrationGPU::CreateTextureObject(T* d_u){
	//cudaArray Descriptor
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

	cudaArray *d_cuArr;
	cudaMemcpy3DParms copyParams = { 0 };


	if (typeid(T) == typeid(float)){
		d_cuArr = d_cuArr_float;
	} 
	else if (typeid(T) == typeid(float4)){
			d_cuArr = d_cuArr_float4;
	}
	else {
		exit(1);
	}

	//Array creation
	copyParams.srcPtr = make_cudaPitchedPtr(
		d_u,
		sizeof(T)*paddedDim[0],
		paddedDim[0],
		paddedDim[1]);

	copyParams.dstArray = d_cuArr;
	copyParams.extent = make_cudaExtent(
		paddedDim[0],
		paddedDim[1],	
		paddedDim[2]);

	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);

	//Array creation End

	cudaResourceDesc    texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_cuArr;
	cudaTextureDesc     texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false;
	if (typeid(T) == typeid(float4)) {
		texDescr.filterMode = cudaFilterModePoint;
		texDescr.filterMode = cudaFilterModeLinear;
	}
	else {
		texDescr.filterMode = cudaFilterModeLinear;
	}
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

		texDescr.readMode = cudaReadModeElementType;
	

	cudaTextureObject_t texObj;
	cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL);
	return texObj;
}

void RegistrationGPU::compose(float4* d_v, float4* d_u) {

	dim3 blocksize = dim3(16, 8, 8);
	dim3 gridsize = dim3(ceil(double(paddedDim[0]) / double(blocksize.x)), ceil(double(paddedDim[1]) / double(blocksize.y)), ceil(double(paddedDim[2]) / double(blocksize.z)));

	unsigned int shsize = sizeof(float4) * blocksize.x * blocksize.y * blocksize.z;

	coordinateImageKernel KERNEL_ARGS2(gridsize, blocksize) (d_v, padDim);
	cudaDeviceSynchronize();

	getCudaError("CreateTexObject-before");
	cudaTextureObject_t d_tex_v = CreateTextureObject<float4>(d_v);
	getCudaError("CreateTexObject");

	if (debug) {
		cout << "composing vector field..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	composeKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_u, d_tex_v, padDim);

	getCudaError("interpolateVectorKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize interpolateVectorKernel");
	cudaMemcpy(d_v, d_u, sizeof(float4)*paddedlen, cudaMemcpyDeviceToDevice);

	cudaDestroyTextureObject(d_tex_v);
	getCudaError("destroyTextureObject");

}

void RegistrationGPU::compose(float4* d_s) {

	dim3 blocksize = dim3(16, 8, 8);
	dim3 gridsize = dim3(ceil(double(paddedDim[0]) / double(blocksize.x)), ceil(double(paddedDim[1]) / double(blocksize.y)), ceil(double(paddedDim[2]) / double(blocksize.z)));

	unsigned int shsize = sizeof(float4) * blocksize.x * blocksize.y * blocksize.z;

	coordinateImageKernel KERNEL_ARGS2(gridsize, blocksize) (d_s, padDim);
	cudaDeviceSynchronize();

	getCudaError("CreateTexObject-before");
	cudaTextureObject_t d_tex_s = CreateTextureObject<float4>(d_s);
	getCudaError("CreateTexObject");

	if (debug) {
		cout << "composing vector field..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	selfComposeKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_s, d_tex_s, padDim);

	getCudaError("interpolateVectorKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize interpolateVectorKernel");

	//cudaResourceDesc h_texResDesc;
	//memset(&h_texResDesc, 0, sizeof(cudaResourceDesc));
	//cudaGetTextureObjectResourceDesc(&h_texResDesc, d_tex_s);
	//cudaFreeArray(h_texResDesc.res.array.array);

	////cudaFree(texDesc);
	////cudaFree(texResDesc);

	cudaDestroyTextureObject(d_tex_s);
	getCudaError("destroyTextureObject");

}


float RegistrationGPU::FindMaxU(float4 *d_u){
	//copy the w of from float4 array d_u to a float array d_uw


	int blocksize = 1024;
	int gridsize = ceil(paddedlen / 1024.0);

	u2uw KERNEL_ARGS2(gridsize, blocksize) (d_u, d_uw, paddedlen);
	getCudaError("u2uw");
	cudaDeviceSynchronize();
	// thrust kicks in
	thrust::device_vector<float> devVec(d_uw, d_uw + paddedlen);
	thrust::device_vector<float>::iterator iter =
		thrust::max_element(devVec.begin(), devVec.end());

	float max_val = *iter;

	return max_val;
}

float RegistrationGPU::FindMaxSx(float4 *d_s){
	//copy the w of from float4 array d_u to a float array d_uw


	int blocksize = 1024;
	int gridsize = ceil(paddedlen / 1024.0);

	u2uw KERNEL_ARGS2(gridsize, blocksize) (d_u, d_uw, paddedlen);
	getCudaError("u2uw");
	cudaDeviceSynchronize();
	// thrust kicks in
	thrust::device_vector<float> devVec(d_uw, d_uw + paddedlen);
	thrust::device_vector<float>::iterator iter =
		thrust::max_element(devVec.begin(), devVec.end());

	float max_val = *iter;

	return max_val;
}

void RegistrationGPU::expfield(float4 *d_v, float4 *d_s)
{
	// Choose N such that 2^-n is close to 0
	float mxnorm = FindMaxU(d_v);
	cout << "largest normalized vector sqaured: " << mxnorm << endl;
	mxnorm = sqrt(mxnorm);
	int N = 0;
	while (mxnorm > 0.5f){
		N++;
		mxnorm *= 0.5;
	}
	float scale = pow((float)2, -N);

	// Launch Kernel to perform explicit first-order integration
	int blocksize = 1024;
	int gridsize = ceil(paddedlen / 1024.0);
	unsigned int shsize = sizeof(float4) * blocksize;
	if (debug){
		cout << "First-order integration..." << endl;
		cout << "blocksize: " << blocksize << " 1 1 " <<  " " << endl;
		cout << "gridsize: " << gridsize << " 1 1 " << " " << endl;
	}
	scaleKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_s, d_v, scale, paddedlen);
	getCudaError("scaleKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize scaleKernel");


	//Recursive scaling and squaring
	printf("self-composing for %i times...\n", N);
	for (int i = 0; i < N; ++i){
		compose(d_s);
		compose_count++;
	}
}

void RegistrationGPU::iminterpolate(float* d_Mp,  float4* d_s){
	
	dim3 blocksize = dim3(8, 8, 8);
	dim3 gridsize = dim3(ceil(double(paddedDim[0]) / double(blocksize.x)), ceil(double(paddedDim[1]) / double(blocksize.y)), ceil(double(paddedDim[2]) / double(blocksize.z)));

	unsigned int shsize = (sizeof(float) + sizeof(float4)) * blocksize.x * blocksize.y * blocksize.z;

	if (debug){
		cout << "interpolating image..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}
	interpolateImageKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_Mp, d_s, d_tex_M, padDim);


	getCudaError("interpolateImageKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize interpolateImageKernel");

}

float RegistrationGPU::energy(float* d_f, float* d_Mp, float4 *d_s){
	
	float reg_weight = (opt.sigma_i*opt.sigma_i) / (opt.sigma_x*opt.sigma_x);
	//pixel-wise energy
	dim3 blocksize = dim3(8, 8, 8);
	dim3 gridsize = dim3(ceil(double(paddedDim[0]) / double(blocksize.x)), ceil(double(paddedDim[1]) / double(blocksize.y)), ceil(double(paddedDim[2]) / double(blocksize.z)));

	if (debug){
		cout << "Computing energy..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}
	energyKernel KERNEL_ARGS2(gridsize, blocksize) (d_f, d_Mp, d_s, d_e, reg_weight, padDim);

	getCudaError("energyKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize energyKernel");

	// thrust kicks in
	thrust::device_vector<float> devVec(d_e, d_e + paddedlen);
	float sum = thrust::reduce(devVec.begin(), devVec.end());

	return sum;
}


float RegistrationGPU::cpuEnergy(float* d_f, float* d_Mp, float4 *d_s){
	cudaMemcpy(fixed, d_f, sizeof(float)*paddedlen, cudaMemcpyDeviceToHost);
	cudaMemcpy(deformedMoving, d_Mp, sizeof(float)*paddedlen, cudaMemcpyDeviceToHost);
	float4* h_s = new float4[paddedlen];
	cudaMemcpy(h_s, d_s, sizeof(float4)*paddedlen, cudaMemcpyDeviceToHost);
	makefloat(h_s, sx, 0);
	makefloat(h_s, sy, 1);
	makefloat(h_s, sz, 2);
	delete[] h_s;

	return Registration::energy(fixed, deformedMoving, sx, sy, sz);
}

void RegistrationGPU::makefloat(float4* h_f4, float* h_f, int col){
	for (int i = 0; i < paddedlen; i++){
		if (col == 0)	h_f[i] = h_f4[i].x;
		if (col == 1)	h_f[i] = h_f4[i].y;
		if (col == 2)	h_f[i] = h_f4[i].z;
		if (col == 3)	h_f[i] = h_f4[i].w;
	}
}


void RegistrationGPU::Savetobin(float4* f4, int col, std::string filename)
{

	float4* h_f4 = new float4[paddedlen];
	cudaMemcpy(h_f4, f4, sizeof(float4)*paddedlen, cudaMemcpyDeviceToHost);
	float* f = new float[paddedlen];
	makefloat(h_f4, f, col);
	saveSlice<float>(f,paddedDim, filename);
	delete[] f;
	delete[] h_f4;
}

void RegistrationGPU::Savetobin(float* f, std::string filename)
{
	float* h_f = new float[paddedlen];
	cudaMemcpy(h_f, f, sizeof(float)*paddedlen, cudaMemcpyDeviceToHost);
	saveSlice<float>(h_f, paddedDim, filename);

	delete[] h_f;
}