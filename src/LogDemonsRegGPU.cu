#include "LogDemonsRegGPU.cuh"

void LogDemonsRegGPU::printDeviceProp(int deviceID){
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

void LogDemonsRegGPU::getCudaError(const char* s){
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

cudaTextureObject_t LogDemonsRegGPU::initGaussValue(float* &h_gaussianVal, float* &d_gaussVal, float sigma, int &radius){

	// Compute Gaussian Values
	radius = int(ceil(3.0f *sigma));
	h_gaussianVal = new float[radius + 1];
	float sum = 0;
	for (int i = 0; i < radius + 1; ++i){
		h_gaussianVal[i] = exp(-(i*i) / (2 * sigma));
		i == 0 ? sum += h_gaussianVal[i] / 2.0 : sum += h_gaussianVal[i];
	}
	for (int i = 0; i < radius + 1; ++i){
		h_gaussianVal[i] /= (2 * sum);
	}

	// copy the gaussian values to the device
	cudaMalloc((void**)&d_gaussVal, sizeof(float)*(radius + 1));
	cudaMemcpy(d_gaussVal, h_gaussianVal, sizeof(float)*(radius + 1), cudaMemcpyHostToDevice);
	getCudaError("CreateGaussianMemory");


	// create texture object for fast access
	cudaResourceDesc h_resDesc;
	memset(&h_resDesc, 0, sizeof(h_resDesc));
	h_resDesc.resType = cudaResourceTypeLinear;
	h_resDesc.res.linear.devPtr = d_gaussVal;
	h_resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	h_resDesc.res.linear.desc.x = 32;
	h_resDesc.res.linear.sizeInBytes = (radius+1)*sizeof(float);
	

	cudaTextureDesc h_texDesc;
	memset(&h_texDesc, 0, sizeof(h_texDesc));
	h_texDesc.readMode = cudaReadModeElementType;
	h_texDesc.filterMode = cudaFilterModePoint;

	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &h_resDesc, &h_texDesc, NULL);
	getCudaError("CreateGaussianTextureObject");

	return tex;
}

cudaTextureObject_t LogDemonsRegGPU::CreateTextureObject(float* d_I, cudaArray *d_cuArr){
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMemcpy3DParms copyParams = { 0 };

	//Copy array form d_I to  cudaArray d_cuArr
	copyParams.srcPtr = make_cudaPitchedPtr(
		d_I,
		sizeof(float)*dim[0],
		dim[0],
		dim[1]);

	copyParams.dstArray = d_cuArr;
	copyParams.extent = make_cudaExtent(
		dim[0],
		dim[1],
		dim[2]);

	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);
	cudaResourceDesc    texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_cuArr;
	cudaTextureDesc     texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;
	cudaTextureObject_t texObj;
	cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL);
	return texObj;
}


void LogDemonsRegGPU::initialize(){
	// initialize the CPU memory first
	LogDemonsReg::initialize();
	d3_dim = dim3(dim[0], dim[1], dim[2]);

	// Initialize GPU memory for images
	cudaMalloc((void**)&d_fixed, sizeof(float)*len);
	cudaMemcpy(d_fixed, fixed, sizeof(float)*len, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_moving, sizeof(float)*len);
	cudaMemcpy(d_moving, moving, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_deformedMoving, sizeof(float)*len);
	cudaMemcpy(d_deformedMoving, deformedMoving, sizeof(float)*len, cudaMemcpyHostToDevice);
	getCudaError("GPU Image initialization");

	// Initialize GPU memory for vector field
	cudaMalloc((void**)&d_ux, sizeof(float)*len);
	cudaMemset(d_ux, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_uy, sizeof(float)*len);
	cudaMemset(d_uy, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_uz, sizeof(float)*len);
	cudaMemset(d_uz, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_vx, sizeof(float)*len);
	cudaMemset(d_vx, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_vy, sizeof(float)*len);
	cudaMemset(d_vy, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_vz, sizeof(float)*len);
	cudaMemset(d_vz, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_sx, sizeof(float)*len);
	cudaMemset(d_sx, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_sy, sizeof(float)*len);
	cudaMemset(d_sy, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_sz, sizeof(float)*len);
	cudaMemset(d_sz, 0, sizeof(float)*len);
	getCudaError("GPU Vector field Memory initialization");

	// Initialize GPU memory for temp memory
	cudaMalloc((void**)&d_uxf, sizeof(float)*len);
	cudaMemset(d_uxf, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_uyf, sizeof(float)*len);
	cudaMemset(d_uyf, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_uzf, sizeof(float)*len);
	cudaMemset(d_uzf, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_en, sizeof(float)*len);
	cudaMemset(d_en, 0, sizeof(float)*len);

	// Initialize GPU 3D Array memory
	cudaChannelFormatDesc channelDescfloat = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&d_cuArr_mov, &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]),	0);
	cudaMalloc3DArray(&d_cuArr_vx,  &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);
	cudaMalloc3DArray(&d_cuArr_vy,  &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);	
	cudaMalloc3DArray(&d_cuArr_vz,  &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);

	// Create moving texture for interpolation
	tex_mov = CreateTextureObject(d_moving, d_cuArr_mov);

	// Precompute fixed image gradient
	gradient(d_fixed, d_uxf, d_uyf, d_uzf);

	// Initialize Guassian Values and put into texture memory
	tex_gauss_f = initGaussValue(gaussian_f, d_gaussian_f, opt.sigma_f, radius_f);
	tex_gauss_d = initGaussValue(gaussian_d, d_gaussian_d, opt.sigma_d, radius_d);

	getCudaError("GPU Gaussian Value initialization");
}

void LogDemonsRegGPU::Register(){
	LogDemonsRegGPU::initialize();

	for (int iter = 0; iter < opt.iteration_max; ++iter) {

		/*	Given the current transformation s, compute a correspondence update field u
		by mimimizing E(u) w.r.t. u		*/
		findupdate();
		cout << "findupdate" << endl;

		/*	For a fluid-like regularization let u <- K(sigma_f) * u	*/
		this->imgaussian(d_ux, d_uy, d_uz, tex_gauss_f, radius_f);
		cout << "imgaussian 0 " << endl;

		/*	Let v <- v compose u	*/
		compose();
		cout << "compose " << endl;

		/*	For a diffusion like regularization let s <- K(sigma_d)*c (else, s<-c)	*/
		imgaussian(d_vx, d_vy, d_vz, tex_gauss_d, radius_d);
		cout << "imgaussian 1 " << endl;

		/*	s = exp(v)	*/
		expfield();
		cout << "expfield " << endl;

		//Transform the moving image
		iminterpolate();
		cout << "iminterpolate " << endl;
		//evulate energy
		energy_vec.push_back(energy());
		cout << "energy " << endl;
		printf("Iteration %i - Energy: %f\n", iter + 1, energy_vec.back());

		if (iter > 4){
			if ((energy_vec[iter - 5] - energy_vec[iter]) < (energy_vec[0] * opt.stop_criterium)){
				printf("e-5: %f\n", energy_vec[iter - 5]);
				printf("e: %f\n", energy_vec[iter]);
				printf("e-5 - e: %f\n", energy_vec[iter - 5] - energy_vec[iter]);
				printf("e[0] * opt.stop_criterium: %f\n", energy_vec[0] * opt.stop_criterium);
				//break;
			}
		}
		std::string filename = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\gpu_results\\Mp_" + to_string(iter + 1) + std::string(".bin");
		saveImage<float>(deformedMoving, dim, filename.c_str());
	}

	printf("LogDemonsRegGPU Complete\n");	printf("\n"); printf("\n");
	for (int iter = 0; iter < energy_vec.size(); ++iter){
		printf("Iteration %i - Energy: %f\n", iter + 1, energy_vec[iter]);
	}
}

void LogDemonsRegGPU::gradient(float* d_I, float* d_fx, float* d_fy, float* d_fz){
	dim3 blocksize = dim3(8, 8, 8);
	dim3 gridsize = dim3(ceil(double(dim[0]) / double(blocksize.x)), ceil(double(dim[1]) / double(blocksize.y)), ceil(double(dim[2]) / double(blocksize.z)));

	if (debug){
		cout << "Gradient decomposing image..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	gradientKernel KERNEL_ARGS2(gridsize, blocksize) (d_I, d_fx, d_fy, d_fz, d3_dim);
	getCudaError("gradientKernel");
	if (debug) {
		cudaDeviceSynchronize();
		getCudaError("DeviceSynchorize gradientKernel");
	}
}

float LogDemonsRegGPU::thrustFindMaxElement(float* d_f){
	// Use thrust to get the max element
	thrust::device_vector<float> dev_vec(d_f, d_f + len);
	thrust::device_vector<float>::iterator iter = thrust::max_element(dev_vec.begin(), dev_vec.end());
	float max_val = *iter;
	return max_val;
}

void LogDemonsRegGPU::findupdate(){
	gradient(d_deformedMoving, d_ux, d_uy, d_uz);
	float alpha2 = (opt.sigma_i*opt.sigma_i) / (opt.sigma_x*opt.sigma_x);

	dim3 blocksize = dim3(1024, 1, 1);
	dim3 gridsize = dim3(ceil(float(len) / float(blocksize.x)), 1, 1);

	if (debug){
		cout << "Computing update kernel..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}
	normalizeVectorKernel KERNEL_ARGS2(gridsize, blocksize) (d_ux, d_uy, d_uz, d_normg2, len);
	getCudaError("normalizeVectorKernel");
	if (debug) {
		cudaDeviceSynchronize();
		getCudaError("DeviceSynchorize normalizeVectorKernel");
	}

	updateKernel KERNEL_ARGS2(gridsize, blocksize) (d_fixed, d_deformedMoving, d_ux, d_uy, d_uz, d_uxf, d_uyf, d_uzf, alpha2, d3_dim);

	getCudaError("updateKernel");
	if (debug) {
		cudaDeviceSynchronize();
		getCudaError("DeviceSynchorize updateKernel");
	}
}

void LogDemonsRegGPU::imgaussian(float* d_fx, float* d_fy, float* d_fz, cudaTextureObject_t tex, int radius){
	
	if (maxdim(marginSize) < radius) printf("Warning: Margin size (%u) is smaller than the gaussian radius (%i). Increase the padding factor. \n", maxdim(marginSize), radius);
		
	//X-direction
	dim3 blocksize;
	dim3 gridsize;
	int shsize;
	
	blocksize=dim3(dim[0], 1, 1);
	gridsize= dim3(dim[1], dim[2],1);
	shsize = sizeof(float4)*(blocksize.x + radius * 2);

	if (debug){
		cout << "Gaussian blur... X direction" << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fx, tex, d3_dim, radius, 0);
	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fy, tex, d3_dim, radius, 0);
	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fz, tex, d3_dim, radius, 0);
	getCudaError("gaussianKernel - X");
	if (debug){
		cudaDeviceSynchronize();
		getCudaError("DeviceSynchorize Gaussian X");
	}
	//Y-direciton
	blocksize = dim3{ 32, 32, 1 };
	gridsize = dim3{ (unsigned int)ceil(dim[0] / 32.0), dim[2], 1 };
	shsize = 32 * dim[1] * sizeof(float);

	if (debug){
		cout << "Gaussian blur... Y direction" << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}

	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fx, tex, d3_dim, radius, 1);
	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fy, tex, d3_dim, radius, 1);
	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fz, tex, d3_dim, radius, 1);
	getCudaError("gaussianKernel - Y");
	if (debug){
		cudaDeviceSynchronize();
		getCudaError("DeviceSynchorize Gaussian Y");
	}
	//Z-direciton
	blocksize = dim3{ 32, 32, 1 };
	gridsize = dim3{ (unsigned int)ceil(dim[0] / 32.0), dim[1], 1 };
	shsize = 32 * dim[2] * sizeof(float);

	if (debug){
		cout << "Gaussian blur... Z direction" << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}
	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fx, tex, d3_dim, radius, 2);
	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fy, tex, d3_dim, radius, 2);
	gaussianKernel KERNEL_ARGS3(gridsize, blocksize, shsize) (d_fz, tex, d3_dim, radius, 2);
	getCudaError("gaussianKernel - Z");
		if (debug){
		cudaDeviceSynchronize();
		getCudaError("DeviceSynchorize Gaussian Z");
	}
}

void LogDemonsRegGPU::compose(){
	dim3 blocksize;
	dim3 gridsize;

	blocksize = dim3(dim[0], 1, 1);
	gridsize = dim3(dim[1], dim[2], 1);
	coordinateImageKernel KERNEL_ARGS2(gridsize, blocksize) (d_vx, d_vy, d_vz, d3_dim, addCoordinate);
	coordinateImageKernel KERNEL_ARGS2(gridsize, blocksize) (d_ux, d_uy, d_uz, d3_dim, addCoordinate);

	tex_vx = CreateTextureObject(d_vx, d_cuArr_vx);
	tex_vy = CreateTextureObject(d_vy, d_cuArr_vx);
	tex_vz = CreateTextureObject(d_vz, d_cuArr_vx);

	interpolate(tex_vx, ux, uy, uz, vx);
	interpolate(tex_vy, ux, uy, uz, vy);
	interpolate(tex_vz, ux, uy, uz, vz);

	coordinateImageKernel KERNEL_ARGS2(gridsize, blocksize) (d_vx, d_vy, d_vz, d3_dim, substractCoordinate);


	cudaDestroyTextureObject(tex_vx);
	getCudaError("destroyTextureObject");
	cudaDestroyTextureObject(tex_vy);
	getCudaError("destroyTextureObject");
	cudaDestroyTextureObject(tex_vz);
	getCudaError("destroyTextureObject");
}

void LogDemonsRegGPU::self_compose(){
	dim3 blocksize;
	dim3 gridsize;

	blocksize = dim3(dim[0], 1, 1);
	gridsize = dim3(dim[1], dim[2], 1);
	coordinateImageKernel KERNEL_ARGS2(gridsize, blocksize) (d_sx, d_sy, d_sz, d3_dim, addCoordinate);

	tex_vx = CreateTextureObject(d_sx, d_cuArr_vx);
	tex_vy = CreateTextureObject(d_sy, d_cuArr_vx);
	tex_vz = CreateTextureObject(d_sz, d_cuArr_vx);

	interpolate(tex_vx, sx, sy, sz, vx);
	interpolate(tex_vy, sx, sy, sz, vy);
	interpolate(tex_vz, sx, sy, sz, vz);

	cudaMemcpy(sx, vx, sizeof(float)*len, cudaMemcpyDeviceToDevice);
	cudaMemcpy(sy, vy, sizeof(float)*len, cudaMemcpyDeviceToDevice);
	cudaMemcpy(sz, vz, sizeof(float)*len, cudaMemcpyDeviceToDevice);

	coordinateImageKernel KERNEL_ARGS2(gridsize, blocksize) (d_sx, d_sy, d_sz, d3_dim, substractCoordinate);

	cudaDestroyTextureObject(tex_vx);
	getCudaError("destroyTextureObject");
	cudaDestroyTextureObject(tex_vy);
	getCudaError("destroyTextureObject");
	cudaDestroyTextureObject(tex_vz);
	getCudaError("destroyTextureObject");
}

void LogDemonsRegGPU::iminterpolate(){
	dim3 blocksize;
	dim3 gridsize;

	blocksize = dim3(dim[0], 1, 1);
	gridsize = dim3(dim[1], dim[2], 1);
	coordinateImageKernel KERNEL_ARGS2(gridsize, blocksize) (d_sx, d_sy, d_sz, d3_dim, addCoordinate);

	interpolate(tex_mov, sx, sy, sz, deformedMoving);
}

void LogDemonsRegGPU::interpolate(cudaTextureObject_t tex_I, float* d_sx, float* d_sy, float* d_sz, float* d_Ip){

	dim3 blocksize = dim3(8, 8, 8);
	dim3 gridsize = dim3(ceil(double(dim[0]) / double(blocksize.x)), ceil(double(dim[1]) / double(blocksize.y)), ceil(double(dim[2]) / double(blocksize.z)));

	if (debug){
		cout << "interpolating image..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}
	interpolateImageKernel KERNEL_ARGS2(gridsize, blocksize) (tex_I, d_sx, d_sy, d_sz, d_Ip, d3_dim);
	
	getCudaError("interpolateImageKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize interpolateImageKernel");
}



void LogDemonsRegGPU::expfield(){
	dim3 blocksize = dim3(1024, 1, 1);
	dim3 gridsize = dim3(ceil(float(len) / float(blocksize.x)), 1, 1);

	if (debug){
		cout << "Computing update kernel..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}
	normalizeVectorKernel KERNEL_ARGS2(gridsize, blocksize) (d_vx, d_vy, d_vz, d_normg2, len);
	getCudaError("normalizeVectorKernel");
	if (debug) {
		cudaDeviceSynchronize();
		getCudaError("DeviceSynchorize normalizeVectorKernel");
	}

	float v2max = thrustFindMaxElement(d_normg2);
	int N = 0; 
	while (v2max > 0.5f){
		N++;
		v2max *= 0.5;
	}
	float scale = pow((float)2, -N);

	if (debug){
		cout << "First-order integration..." << endl;
		cout << "blocksize: " << blocksize.x << " 1 1 " << " " << endl;
		cout << "gridsize: " << gridsize.x << " 1 1 " << " " << endl;
	}

	scaleKernel KERNEL_ARGS2(gridsize, blocksize) (d_vx, d_vy, d_vz, d_sx, d_sy, d_sz, scale, len);

	//Recursive scaling and squaring
	printf("self-composing for %i times...\n", N);
	for (int i = 0; i < N; ++i){
		self_compose();
	}
}


float LogDemonsRegGPU::energy(){

	float reg_weight = (opt.sigma_i*opt.sigma_i) / (opt.sigma_x*opt.sigma_x);
	//pixel-wise energy
	dim3 blocksize = dim3(8, 8, 8);
	dim3 gridsize = dim3(ceil(double(dim[0]) / double(blocksize.x)), ceil(double(dim[1]) / double(blocksize.y)), ceil(double(dim[2]) / double(blocksize.z)));

	if (debug){
		cout << "Computing energy..." << endl;
		cout << "blocksize: " << blocksize.x << " " << blocksize.y << " " << blocksize.z << " " << endl;
		cout << "gridsize: " << gridsize.x << " " << gridsize.y << " " << gridsize.z << " " << endl;
	}
	energyKernel KERNEL_ARGS2(gridsize, blocksize) (d_fixed, d_deformedMoving, d_sx, d_sy, d_sz, d_en, reg_weight, d3_dim);

	getCudaError("energyKernel");
	cudaDeviceSynchronize();
	getCudaError("DeviceSynchorize energyKernel");

	// thrust kicks in
	thrust::device_vector<float> devVec(d_en, d_en + len);
	float sum = thrust::reduce(devVec.begin(), devVec.end());

	return sum;
}