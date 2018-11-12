//Std lib include
#include <stdio.h>
#include <cstdlib>
#include <fstream>

//Custom include
#include "LogDemonsReg.cuh"
#include "LogDemonsReg_testfun.cuh"
#include "LogDemonsRegGPU.cuh"
#include "RegistrationIO.h"
//#include "RegistrationGPU.cuh"

#define TEST 0

void runtestfunctions();

int main(int argc, char* argv[])
{
	if (TEST) {
		runtestfunctions();
		std::cout << "Program exits, Press enter...";

		getchar();
		return 0;
	}

	float *fixedData, *movingData;
	unsigned int dim[3];
	const char* FixPath = "..\\test_data\\FixedBrain_1.dat";
	const char* MovPath = "..\\test_data\\MovingBrain_1.dat";

	//Load data
	FILE *file = fopen(FixPath, "rb");
	short sdim[3];
	fread(&sdim, sizeof(short), 3, file);
	int len = sdim[0] * sdim[1] * sdim[2];
	dim[0] = (unsigned int)sdim[0];
	dim[1] = (unsigned int)sdim[1];
	dim[2] = (unsigned int)sdim[2];

	cout << "Reading file with length of: " << len << endl;

	fixedData = new float[len];
	fread(fixedData, sizeof(float), len, file);
	fclose(file);

	file = fopen(MovPath, "rb");
	fread(&sdim, sizeof(short), 3, file);
	len = sdim[0] * sdim[1] * sdim[2];
	cout << "Reading file with length of: " << len << endl;
	movingData = new float[len];
	fread(movingData, sizeof(float), len, file);
	fclose(file);

	//normalize both images to 0-255 scale
	float fmin = 10000, fmax = 0, mmin = 10000, mmax = 0;
	for (int i = 0; i < len; ++i){
		if (fixedData[i] < fmin) fmin = fixedData[i];
		if (fixedData[i] > fmax) fmax = fixedData[i];
		if (movingData[i] < mmin) mmin = movingData[i];
		if (movingData[i] > mmax) mmax = movingData[i];
	}
	for (int i = 0; i < len; ++i){
		fixedData[i] = 255*(fixedData[i] - fmin) / (fmax - fmin);
		movingData[i] = 255 * (movingData[i] - mmin) / (mmax - mmin);
	}

	LogDemonsRegGPU reg(fixedData, movingData, dim);
	reg.Register();

	//Cleanup
	delete[] fixedData, movingData;


	std::cout << "Program exits, Press enter...";

	getchar();
	return 0;
}

void runtestfunctions(){

	// I did not set up garbage collection for the test functions. Use with caution.
	LogDemonsReg_testfun test;
	//test._findupdate();
	//test._imgaussian(3);
	//test._iminterpolate();
	//test._compose();
	//test._expfield();
	//test._energy();

}