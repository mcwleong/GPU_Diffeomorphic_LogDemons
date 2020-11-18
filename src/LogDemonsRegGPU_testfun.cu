#include "LogDemonsRegGPU_testfun.cuh"

void LogDemonsRegGPU_testfun::_findupdate(){
	/*
	Input: fixed, deformedMoving
	Memory accessed: ux, uy, uz, uxf, uyf, uzf, normg2, normg2f;
	Output: velocity update field in ux, uy, uz
	*/

	const char* FixPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\data\\F.dat";
	const char* MovPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\data\\Mp.dat";

	// Load data
	// Define dimension, which directly access the variable "dim" and "len" on the base class
	// Assumes both bin file shares the same dimension
	short sdim[3];
	FILE *file = fopen(FixPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];
	cout << "Reading file with length of: " << len << endl;
	fclose(file);

	// Allocate memory
	fixed = new float[len]();
	deformedMoving = new float[len]();
	ux = new float[len]();
	uy = new float[len]();
	uz = new float[len]();
	uxf = new float[len]();
	uyf = new float[len]();
	uzf = new float[len]();
	normg2 = new float[len]();
	normg2f = new float[len]();

	// Read data
	file = fopen(FixPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(fixed, sizeof(float), len, file);
	fclose(file);

	file = fopen(MovPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(deformedMoving, sizeof(float), len, file);
	fclose(file);

	LogDemonsReg::gradient(fixed, uxf, uyf, uzf, normg2f);

	// Allocate GPU Memory
	d3_dim = dim3(dim[0], dim[1], dim[2]);
 
	cudaMalloc((void**)&d_fixed, sizeof(float)*len);
	cudaMemcpy(d_fixed, fixed, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_deformedMoving, sizeof(float)*len);
	cudaMemcpy(d_deformedMoving, deformedMoving, sizeof(float)*len, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_uxf, sizeof(float)*len);
	cudaMemcpy(d_uxf, uxf, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_uyf, sizeof(float)*len);
	cudaMemcpy(d_uyf, uyf, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_uzf, sizeof(float)*len);
	cudaMemcpy(d_uzf, uzf, sizeof(float)*len, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_ux, sizeof(float)*len);
	cudaMemset(d_ux, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_uy, sizeof(float)*len);
	cudaMemset(d_uy, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_uz, sizeof(float)*len);
	cudaMemset(d_uz, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_normg2, sizeof(float)*len);
	cudaMemset(d_normg2, 0, sizeof(float)*len);


	// Compute update field using GPU
	findupdate();

	// Fetch results from GPU
	cudaMemcpy(ux, d_ux, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, d_uy, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(uz, d_uz, sizeof(float)*len, cudaMemcpyDeviceToHost);

	
	saveImage<float>(ux, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\gpu_results\\ux_gpu.bin");
	saveImage<float>(uy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\gpu_results\\uy_gpu.bin");
	saveImage<float>(uz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\gpu_results\\uz_gpu.bin");
}


void LogDemonsRegGPU_testfun::_gradient(){
	/*
	Input: fixed
	Output: velocity update field in uxf, uyf, uzf
	*/

	const char* FixPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\data\\F.dat";


	// Load data
	// Define dimension, which directly access the variable "dim" and "len" on the base class
	// Assumes both bin file shares the same dimension
	short sdim[3];
	FILE *file = fopen(FixPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];
	cout << "Reading file with length of: " << len << endl;
	fclose(file);

	// Allocate memory
	fixed = new float[len]();
	deformedMoving = new float[len]();
	uxf = new float[len]();
	uyf = new float[len]();
	uzf = new float[len]();

	// Read data
	file = fopen(FixPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(fixed, sizeof(float), len, file);
	fclose(file);


	// Allocate GPU Memory
	d3_dim = dim3(dim[0], dim[1], dim[2]);

	cudaMalloc((void**)&d_fixed, sizeof(float)*len);
	cudaMemcpy(d_fixed, fixed, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_ux, sizeof(float)*len);
	cudaMemset(d_ux, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_uy, sizeof(float)*len);
	cudaMemset(d_uy, 0, sizeof(float)*len);
	cudaMalloc((void**)&d_uz, sizeof(float)*len);
	cudaMemset(d_uz, 0, sizeof(float)*len);
	gradient(d_fixed, d_ux, d_uy, d_uz);

	// Fetch results from GPU
	cudaMemcpy(uxf, d_ux, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(uyf, d_uy, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(uzf, d_uz, sizeof(float)*len, cudaMemcpyDeviceToHost);

	saveImage<float>(uxf, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\gradient\\gpu_results\\ux_gpu.bin");
	saveImage<float>(uyf, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\gradient\\gpu_results\\uy_gpu.bin");
	saveImage<float>(uzf, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\gradient\\gpu_results\\uz_gpu.bin");
}




void LogDemonsRegGPU_testfun::_imgaussian(float sigma){
	/*
	Input: ux, uy, uz, opt.sigma
	Output: Blurred vector field in ux, uy, uz
	*/
	opt.sigma_f = sigma;

	const char* uxPath = "D:\\GPU_Diffeomorphic_Demons_private\\test_data\\imguassian\\data\\ux.dat";
	const char* uyPath = "D:\\GPU_Diffeomorphic_Demons_private\\test_data\\imguassian\\data\\uy.dat";
	const char* uzPath = "D:\\GPU_Diffeomorphic_Demons_private\\test_data\\imguassian\\data\\uz.dat";
	// Load data
	// Define dimension, which directly access the variable "dim" and "len" on the base class
	// Assumes both bin file shares the same dimension
	short sdim[3];
	FILE *file = fopen(uxPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];
	cout << "Reading file with length of: " << len << endl;
	fclose(file);

	// Allocate memory
	moving = new float[len]();
	deformedMoving = new float[len]();
	ux = new float[len]();
	uy = new float[len]();
	uz = new float[len]();


	//load data
	file = fopen(uxPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(ux, sizeof(float), len, file);
	fclose(file);

	file = fopen(uyPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(uy, sizeof(float), len, file);
	fclose(file);

	file = fopen(uzPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(uz, sizeof(float), len, file);
	fclose(file);

	// Allocate GPU Memory
	d3_dim = dim3(dim[0], dim[1], dim[2]);
	cudaMalloc((void**)&d_ux, sizeof(float)*len);
	cudaMemcpy(d_ux, ux, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_uy, sizeof(float)*len);
	cudaMemcpy(d_uy, uy, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_uz, sizeof(float)*len);
	cudaMemcpy(d_uz, uz, sizeof(float)*len, cudaMemcpyHostToDevice);

	getCudaError("GPU Memory Allocation");
	//Allocate texture resources
	tex_gauss_d=initGaussValue(gaussian_d, d_gaussian_d, sigma, radius_d);

	// Compute gaussian blur using GPU
	imgaussian(d_ux, d_uy, d_uz, tex_gauss_d, radius_d);
	
	cudaDeviceSynchronize();
	//imgaussian(d_ux, d_uy, d_uz, tex_gauss_d, radius_d);
	// Fetch results from GPU
	cudaMemcpy(ux, d_ux, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, d_uy, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(uz, d_uz, sizeof(float)*len, cudaMemcpyDeviceToHost);

	saveImage<float>(ux, dim, "D:\\GPU_Diffeomorphic_Demons_private\\test_data\\imguassian\\GPU_Results\\uxg_gpu.bin");
	saveImage<float>(uy, dim, "D:\\GPU_Diffeomorphic_Demons_private\\test_data\\imguassian\\GPU_Results\\uyg_gpu.bin");
	saveImage<float>(uz, dim, "D:\\GPU_Diffeomorphic_Demons_private\\test_data\\imguassian\\GPU_Results\\uzg_gpu.bin");

}

void LogDemonsRegGPU_testfun::_iminterpolate(){
	/*
	Input: moving, sx, sy, sz
	Memory accessed: deformedMoving
	Output: Warped image in deformedMoving
	*/

	const char* MovPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\iminterpolate\\data\\M.dat";
	const char* sxPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\iminterpolate\\data\\sy.dat";
	const char* syPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\iminterpolate\\data\\sx.dat";
	const char* szPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\iminterpolate\\data\\sz.dat";

	// Load data
	// Define dimension, which directly access the variable "dim" and "len" on the base class
	// Assumes both bin file shares the same dimension
	short sdim[3];
	FILE *file = fopen(MovPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];
	cout << "Reading file with length of: " << len << endl;
	fclose(file);

	// Allocate memory
	moving = new float[len]();
	deformedMoving = new float[len]();
	sx = new float[len]();
	sy = new float[len]();
	sz = new float[len]();

	// Read data
	file = fopen(MovPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(moving, sizeof(float), len, file);
	fclose(file);

	file = fopen(sxPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sx, sizeof(float), len, file);
	fclose(file);

	file = fopen(syPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sy, sizeof(float), len, file);
	fclose(file);

	file = fopen(szPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sz, sizeof(float), len, file);
	fclose(file);
	
	// Allocate GPU Memory
	d3_dim = dim3(dim[0], dim[1], dim[2]);
	cudaMalloc((void**)&d_moving, sizeof(float)*len);
	cudaMemcpy(d_moving, moving, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sx, sizeof(float)*len);
	cudaMemcpy(d_sx, sx, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sy, sizeof(float)*len);
	cudaMemcpy(d_sy, sy, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sz, sizeof(float)*len);
	cudaMemcpy(d_sz, sz, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_deformedMoving, sizeof(float)*len);
	cudaMemset(d_deformedMoving, 0, sizeof(float)*len);

	// Initialize moving Image texture
	cudaChannelFormatDesc channelDescfloat = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&d_cuArr_mov, &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);
	tex_mov = CreateTextureObject(d_moving, d_cuArr_mov);

	iminterpolate();

	cudaMemcpy(deformedMoving, d_deformedMoving, sizeof(float)*len, cudaMemcpyDeviceToHost);

	saveImage<float>(deformedMoving, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\iminterpolate\\gpu_results\\Mp_gpu.bin");

}

void LogDemonsRegGPU_testfun::_compose(){
	/*
	Input: ux, uy, uz, vx, vy, vz
	Memory accessed: x_p, y_p, z_p
	Output: composed field in vx vy vz
	*/
	
	const char* uxPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\data\\ux.dat";
	const char* uyPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\data\\uy.dat";
	const char* uzPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\data\\uz.dat";
	const char* vxPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\data\\vx.dat";
	const char* vyPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\data\\vy.dat";
	const char* vzPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\data\\vz.dat";

	// Load data
	// Define dimension, which directly access the variable "dim" and "len" on the base class
	// Assumes both bin file shares the same dimension
	short sdim[3];
	FILE *file = fopen(uxPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];
	cout << "Reading file with length of: " << len << endl;
	fclose(file);

	// Allocate memory
	vx = new float[len]();
	vy = new float[len]();
	vz = new float[len]();
	ux = new float[len]();
	uy = new float[len]();
	uz = new float[len]();
	x_p = new float[len]();
	y_p = new float[len]();
	z_p = new float[len]();

	// load data
	file = fopen(uxPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(ux, sizeof(float), len, file);
	fclose(file);

	file = fopen(uyPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(uy, sizeof(float), len, file);
	fclose(file);

	file = fopen(uzPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(uz, sizeof(float), len, file);
	fclose(file);

	file = fopen(vxPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(vx, sizeof(float), len, file);
	fclose(file);

	file = fopen(vyPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(vy, sizeof(float), len, file);
	fclose(file);

	file = fopen(vzPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(vz, sizeof(float), len, file);
	fclose(file);


	// Allocate GPU Memory
	d3_dim = dim3(dim[0], dim[1], dim[2]);
	cudaMalloc((void**)&d_ux, sizeof(float)*len);
	cudaMemcpy(d_ux, ux, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_uy, sizeof(float)*len);
	cudaMemcpy(d_uy, uy, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_uz, sizeof(float)*len);
	cudaMemcpy(d_uz, uz, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_vx, sizeof(float)*len);
	cudaMemcpy(d_vx, vx, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_vy, sizeof(float)*len);
	cudaMemcpy(d_vy, vy, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_vz, sizeof(float)*len);
	cudaMemcpy(d_vz, vz, sizeof(float)*len, cudaMemcpyHostToDevice);

	cudaChannelFormatDesc channelDescfloat = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&d_cuArr_vx, &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);
	cudaMalloc3DArray(&d_cuArr_vy, &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);
	cudaMalloc3DArray(&d_cuArr_vz, &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);

	compose();

	cudaMemcpy(vx, d_vx, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(vy, d_vy, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(vz, d_vz, sizeof(float)*len, cudaMemcpyDeviceToHost);

	// save results
	saveImage<float>(vx, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\gpu_results\\vx_gpu.bin");
	saveImage<float>(vy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\gpu_results\\vy_gpu.bin");
	saveImage<float>(vz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\gpu_results\\vz_gpu.bin");
}

void LogDemonsRegGPU_testfun::_expfield(){
	/*
	Input: vx, vy, vz
	Memory accessed: x_p, y_p, z_p
	Output: sx, sy, sz
	*/
	const char* vxPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\data\\vxg.dat";
	const char* vyPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\data\\vyg.dat";
	const char* vzPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\data\\vzg.dat";

	// Load data
	// Define dimension, which directly access the variable "dim" and "len" on the base class
	// Assumes both bin file shares the same dimension
	short sdim[3];
	FILE *file = fopen(vxPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];
	cout << "Reading file with length of: " << len << endl;
	fclose(file);

	// Allocate memory
	vx = new float[len]();
	vy = new float[len]();
	vz = new float[len]();
	sx = new float[len]();
	sy = new float[len]();
	sz = new float[len]();
	x_p = new float[len]();
	y_p = new float[len]();
	z_p = new float[len]();

	// Load data
	file = fopen(vxPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(vx, sizeof(float), len, file);
	fclose(file);

	file = fopen(vyPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(vy, sizeof(float), len, file);
	fclose(file);

	file = fopen(vzPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(vz, sizeof(float), len, file);
	fclose(file);


	// Allocate GPU Memory
	d3_dim = dim3(dim[0], dim[1], dim[2]);
	cudaMalloc((void**)&d_vx, sizeof(float)*len);
	cudaMemcpy(d_vx, vx, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_vy, sizeof(float)*len);
	cudaMemcpy(d_vy, vy, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_vz, sizeof(float)*len);
	cudaMemcpy(d_vz, vz, sizeof(float)*len, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_sx, sizeof(float)*len);
	cudaMemcpy(d_sx, sx, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sy, sizeof(float)*len);
	cudaMemcpy(d_sy, sy, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sz, sizeof(float)*len);
	cudaMemcpy(d_sz, sz, sizeof(float)*len, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_normg2, sizeof(float)*len);
	cudaMemset(d_normg2, 0, sizeof(float)*len);
	
	cudaChannelFormatDesc channelDescfloat = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&d_cuArr_vx, &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);
	cudaMalloc3DArray(&d_cuArr_vy, &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);
	cudaMalloc3DArray(&d_cuArr_vz, &channelDescfloat, make_cudaExtent(dim[0], dim[1], dim[2]), 0);

	expfield();


	cudaMemcpy(sx, d_sx, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(sy, d_sy, sizeof(float)*len, cudaMemcpyDeviceToHost);
	cudaMemcpy(sz, d_sz, sizeof(float)*len, cudaMemcpyDeviceToHost);

	saveImage<float>(sx, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\gpu_results\\sx_gpu.bin");
	saveImage<float>(sy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\gpu_results\\sy_gpu.bin");
	saveImage<float>(sz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\gpu_results\\sz_gpu.bin");
}


void LogDemonsRegGPU_testfun::_jacobian(){
	/*
	Input: sx, sy, sz, F, Mp
	Memory accessed: gx_x, gx_y, gx_z, gy_x, gy_y, gy_z, gz_x, gz_y, gz_z, jac
	Output: sx, sy, sz
	*/

	const char* sxPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\jacobian\\data\\sx.dat";
	const char* syPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\jacobian\\data\\sy.dat";
	const char* szPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\jacobian\\data\\sz.dat";

	short sdim[3];
	FILE *file = fopen(sxPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];
	cout << "Reading file with length of: " << len << endl;
	fclose(file);

	// Allocate memory
	sx = new float[len]();
	sy = new float[len]();
	sz = new float[len]();
	jac = new float[len]();


	file = fopen(sxPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sx, sizeof(float), len, file);
	fclose(file);

	file = fopen(syPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sy, sizeof(float), len, file);
	fclose(file);

	file = fopen(szPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sz, sizeof(float), len, file);
	fclose(file);


	// Allocate GPU Memory
	d3_dim = dim3(dim[0], dim[1], dim[2]);
	cudaMalloc((void**)&d_sx, sizeof(float)*len);
	cudaMemcpy(d_sx, sx, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sy, sizeof(float)*len);
	cudaMemcpy(d_sy, sy, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sz, sizeof(float)*len);
	cudaMemcpy(d_sz, sz, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_en, sizeof(float)*len);
	cudaMemset(d_en, 0, sizeof(float)*len);

	jacobian();
	
	cudaMemcpy(jac, d_en, sizeof(float)*len, cudaMemcpyDeviceToHost);

	saveImage<float>(jac, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\jacobian\\gpu_results\\jac_gpu.bin");
	
	//printf("%f", e);
	
}


void LogDemonsRegGPU_testfun::_energy(){
	/*
	Input: sx, sy, sz, F, Mp
	Memory accessed: gx_x, gx_y, gx_z, gy_x, gy_y, gy_z, gz_x, gz_y, gz_z, jac
	Output: sx, sy, sz
	*/

	const char* FixPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\F.dat";
	const char* MovPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\Mp.dat";
	const char* sxPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\sx.dat";
	const char* syPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\sy.dat";
	const char* szPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\sz.dat";

	short sdim[3];
	FILE *file = fopen(sxPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];
	cout << "Reading file with length of: " << len << endl;
	fclose(file);

	// Allocate memory
	fixed = new float[len]();
	deformedMoving = new float[len]();
	sx = new float[len]();
	sy = new float[len]();
	sz = new float[len]();
	jac = new float[len]();
	

	file = fopen(FixPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(fixed, sizeof(float), len, file);
	fclose(file);

	file = fopen(MovPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(deformedMoving, sizeof(float), len, file);
	fclose(file);

	file = fopen(sxPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sx, sizeof(float), len, file);
	fclose(file);

	file = fopen(syPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sy, sizeof(float), len, file);
	fclose(file);

	file = fopen(szPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(sz, sizeof(float), len, file);
	fclose(file);


	// Allocate GPU Memory
	d3_dim = dim3(dim[0], dim[1], dim[2]);
	cudaMalloc((void**)&d_fixed, sizeof(float)*len);
	cudaMemcpy(d_fixed, fixed, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_deformedMoving, sizeof(float)*len);
	cudaMemcpy(d_deformedMoving, deformedMoving, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sx, sizeof(float)*len);
	cudaMemcpy(d_sx, sx, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sy, sizeof(float)*len);
	cudaMemcpy(d_sy, sy, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_sz, sizeof(float)*len);
	cudaMemcpy(d_sz, sz, sizeof(float)*len, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_en, sizeof(float)*len);
	cudaMemset(d_en, 0, sizeof(float)*len);

	energy();

	cudaMemcpy(jac, d_en, sizeof(float)*len, cudaMemcpyDeviceToHost);

	saveImage<float>(jac, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\jacobian\\gpu_results\\jac_gpu.bin");

	//printf("%f", e);

}