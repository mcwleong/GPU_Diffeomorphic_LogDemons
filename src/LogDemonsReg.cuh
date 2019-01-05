#pragma once

#include <iostream>
#include "RegistrationIO.h"
#include <vector>

class LogDemonsReg
{
public:
	struct RegistrationOption{
		float sigma_f = 3.0f, sigma_d = 2.0f, sigma_i = 1.0f, sigma_x = 1.0f;
		float pad = 1.2f;
		float stop_criterium = 0.004f;
		int iteration_max = 10;
	} opt;

	//Constructor
	LogDemonsReg(){};
	LogDemonsReg(float *fixed, float *moving, unsigned int ImgSize[3]);

	// Destructor
	~LogDemonsReg();

	// Public functions
	void Register();
	void setImages(float* fixed, float* moving, unsigned int ImgSize[3]);
	void debugOn(bool a) { debug = a; }

protected:
	// Initialization code is set protected as the CUDA version
	// also demand initialization on the host memory
	void initialize();

	// Variables for registration
	// Image source 
	struct ImgSrc{
		float* fixed;
		float* moving;
		unsigned int srcDim[3];
	};
	ImgSrc src;


	bool initialized = false;
	bool debug = true;

	// Image dimension
	unsigned int dim[3], marginSize[3], len = 1;

	// Images for registration
	float *fixed, *moving, *deformedMoving;

	// vector fields: update/velocity/deformation
	float *ux, *uy, *uz, *vx, *vy, *vz, *sx, *sy, *sz, *normg2, *det_J;

	float *uxf, *uyf, *uzf, *normg2f; //Fixed image gradient
	float *gx_x, *gx_y, *gx_z, *gy_x, *gy_y, *gy_z, *gz_x, *gz_y, *gz_z;
	float *jac;

	// temp variables required for computation
	float *x_p, *y_p, *z_p; //required for composition and gaussian smoothing

	//registration energies
	float e_min;
	vector<float> energy_vec;


	// CPU codes
	//get array idx from position ijk
	inline unsigned int pos(int i, int j, int k) {
		return  i + j*dim[0] + k*dim[0] * dim[1];
	}

	//get sign of the value (-1 = negative; 0 = zero; +1 = positive)
	inline int sign(float x) {
		return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
	}

	void padImage();

	void findupdate();
	void gradient(float *I, float* gx, float* gy, float* gz, float* norm);
	void gradient(float *I, float *gx, float *gy, float *gz);

	void compose();
	void self_compose();
	void interp3(float* const ux, float* const uy, float* const uz,
		const float qx, const float qy, const float qz,
		float* const uxp, float* const uyp, float* const uzp);
	void interp3(float* const I, const float qx, const float qy,
		const float qz, float* const Ip);
	void iminterpolate(float* I, float* sx, float* sy, float* sz, float* Ip);
	void expfield();

	void imgaussian(float* fx, float* fy, float* fz, float sigma);
	float energy();
	void jacobian(float* det_J);

};
