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

	// Precompute fixed image gradient
	//gradient(fixed, uxf, uyf, uzf, normg2f);

	//findupdate();
	
	saveImage<float>(ux, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\cpu_results\\ux.bin");
	saveImage<float>(uy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\cpu_results\\uy.bin");
	saveImage<float>(uz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\findupdate\\cpu_results\\uz.bin");
}

void LogDemonsRegGPU_testfun::_imgaussian(float sigma){
	/*
	Input: ux, uy, uz, opt.sigma
	Memory accessed: x_p, y_p, z_p
	Output: Blurred vector field in ux, uy, uz
	*/
	opt.sigma_f = sigma;

	const char* uxPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\imguassian\\data\\ux.dat";
	const char* uyPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\imguassian\\data\\uy.dat";
	const char* uzPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\imguassian\\data\\uz.dat";

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
	x_p = new float[len]();
	y_p = new float[len]();
	z_p = new float[len]();

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

	//imgaussian(ux, uy, uz, opt.sigma_f);
	
	saveImage<float>(ux, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\imguassian\\cpu_results\\uxg.bin");
	saveImage<float>(uy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\imguassian\\cpu_results\\uyg.bin");
	saveImage<float>(uz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\imguassian\\cpu_results\\uzg.bin");


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

	//iminterpolate(moving, sx, sy, sz, deformedMoving);
	
	saveImage<float>(deformedMoving, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\iminterpolate\\cpu_results\\Mp.bin");

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

	// run function to be tested
	//compose();


	// save results
	saveImage<float>(vx, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\cpu_results\\vx.bin");
	saveImage<float>(vy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\cpu_results\\vy.bin");
	saveImage<float>(vz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\compose\\cpu_results\\vz.bin");
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

	// Run function to be tested
	//expfield();

	saveImage<float>(sx, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\cpu_results\\sx.bin");
	saveImage<float>(sy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\cpu_results\\sy.bin");
	saveImage<float>(sz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\expfield\\cpu_results\\sz.bin");
}


void LogDemonsRegGPU_testfun::_energy(){
	/*
	Input: sx, sy, sz, F, Mp
	Memory accessed: gx_x, gx_y, gx_z, gy_x, gy_y, gy_z, gz_x, gz_y, gz_z, jac
	Output: sx, sy, sz
	*/

	const char* sxPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\sx.dat";
	const char* syPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\sy.dat";
	const char* szPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\sz.dat";
	const char* FixPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\F.dat";
	const char* MpPath = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\data\\Mp.dat";

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
	gx_x = new float[len]();
	gx_y = new float[len]();
	gx_z = new float[len]();
	gy_x = new float[len]();
	gy_y = new float[len]();
	gy_z = new float[len]();
	gz_x = new float[len]();
	gz_y = new float[len]();
	gz_z = new float[len]();

	file = fopen(FixPath, "rb");
	fread(&sdim, sizeof(short), 3, file); //discard the dimension header
	fread(fixed, sizeof(float), len, file);
	fclose(file);

	file = fopen(MpPath, "rb");
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

	jac = new float[len]();
	//jacobian(jac);
	
	saveImage<float>(jac, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\energy\\cpu_results\\jac.bin");
	
	//printf("%f", e);
	
}