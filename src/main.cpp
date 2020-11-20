
#include "wrapper.h"
#include "LogDemonsReg.cuh"
#include "LogDemonsReg_testfun.cuh"
#include "LogDemonsRegGPU.cuh"
#include "LogDemonsRegGPU_testfun.cuh"
#include "RegistrationIO.h"
void runtestfunctions() {

	// I did not set up RAM cleanup for the test functions. Use with caution.
	printf("Testfun\n\n");
	LogDemonsReg_testfun test;
	//test._gradient();
	//test._findupdate();
	//test._imgaussian(3);
	//test._iminterpolate();
	//test._compose();
	//test._expfield();
	//test._jacobian();
	//test._energy();

	printf("testfungpu");
	LogDemonsRegGPU_testfun GPUtest;
	GPUtest.debugOn(0);
	GPUtest._gradient();
	//GPUtest._findupdate();
	//GPUtest._imgaussian(3);
	//GPUtest._iminterpolate();
	//GPUtest._compose();
	//GPUtest._jacobian();
	//GPUtest._expfield();
	//GPUtest._energy();
}

  int main()
  {
    //using ImageType = itk::Image< unsigned short, 3 >;
  
    std::cout << "ITK Hello World !" << std::endl;
  
	float *fixedData, *movingData;
	unsigned int dim[3];
	
	
// Windows
//	const char* FixPathdcm = "..\\data\\FixedBrain_1.dcm";
//	const char* MovPathdcm = "..\\data\\MovingBrain_1.dcm";

// Linux
	const char* FixPathdcm = "../data/FixedBrain_1.dcm";
	const char* MovPathdcm = "../data/MovingBrain_1.dcm";
	
	readDicomToBuffer(FixPathdcm, fixedData, dim);
	readDicomToBuffer(MovPathdcm, movingData, dim);

	LogDemonsRegGPU reg(fixedData, movingData, dim);
	reg.debugOn(0);
	//reg.LogDemonsReg::Register();
	reg.opt.iteration_max = 50;
	reg.Register();

	reg.syncGPUMemory();


	//Cleanup
	delete[] fixedData, movingData;


	std::cout << "Program exits, Press enter...";

    return EXIT_SUCCESS;
  }