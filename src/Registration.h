#pragma once
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <chrono>
#include "RegistrationIO.h"
#include <sstream>
using namespace std;
using namespace std::chrono;

class Registration
{
public:
	struct option{
		float sigma_f = 3.0f, sigma_d = 1.0f, sigma_i = 1.0f, sigma_x = 1.0f;
		float pad = 1.0f;
		float stop_criterium = 0.004f;
		int iteration_max = 20;
		int scale = 1;
		char* output_header = "result_";
	};

	Registration();
	Registration(float *fixed, float *moving, unsigned int dimension[3]);
	~Registration();

	virtual void Register();
	void debugOn(bool dbg);
	void setImages(float& fixed, float& moving, unsigned int dimension[3]);
	template <typename T> int sign(T val);

	option opt;
	friend void getImg(Registration* reg, int i);

	//Shared among CPU and GPU version of the code
protected:
	void initialize();
	void initialize(int scale);

	struct SourcePtr{ 
		//Source ptr, defined as const ptr at the constructor
		SourcePtr(float * ptr) : data(ptr){}
		float * const data;

	};

	SourcePtr *fixedsrc, *movingsrc;
	bool loaded = false, initialized = false,  debug = false;
	unsigned int dim[3], marginSize[3], paddedDim[3], paddedlen = 1;
	float *fixed, *moving, *deformedMoving;
	float *ux, *uy, *uz, *vx, *vy, *vz,*sx, *sy, *sz;
	float *e, e_min;
	
	void saveResult(float* Mp, float* sx, float* sy, float* sz);

	//CPU codes only run-able on CPU
//private:
	unsigned int pos(int i, int j, int k);
	void padImage();

	void findupdate(float* F, float* Mp, float* ux, float* uy, float* uz);
	void gradient(float *I, float* gx, float* gy, float* gz, float* normg2);
	void gradient(float *I, float *gx, float *gy, float *gz);
	
	void compose(float* vx, float* vy, float* vz, float* ux, float* uy, float* uz);
	void compose(float* vx, float* vy, float* vz);
	void interp3(float* const ux, float* const uy, float* const uz, 
		const float qx, const float qy, const float qz, 
		float* const uxp, float* const uyp, float* const uzp);
	void interp3(float* const I, const float qx, const float qy,
		const float qz, float* const Ip);
	void iminterpolate(float* I, float* sx, float* sy, float* sz, float* Ip);
	void expfield(float* ux, float* uy, float* uz, float* sx, float* sy, float* sz);
	
	void imgaussian(float* ux, float* uy, float* uz, float sigma);
	float energy(float* F, float* Mp, float* sx, float* sy, float* sz);
	void jacobian(float* det_J, float* sx, float* sy, float* sz);


	int compose_count = 0;
};
