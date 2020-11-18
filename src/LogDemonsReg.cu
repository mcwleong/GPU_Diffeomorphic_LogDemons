#include "LogDemonsReg.cuh"

//Public Functions

//Default constructor

LogDemonsReg::LogDemonsReg(float *F, float *M, unsigned int dimension[3])
{
	src.fixed = F;
	src.moving = M;
	src.srcDim[0] = dimension[0];
	src.srcDim[1] = dimension[1];
	src.srcDim[2] = dimension[2];
}

LogDemonsReg::~LogDemonsReg()
{
	if (initialized){
		delete[] fixed;
		delete[] moving;
		delete[] deformedMoving;
		delete[] ux;
		delete[] uy;
		delete[] uz;
		delete[] vx;
		delete[] vy;
		delete[] vz;
		delete[] sx;
		delete[] sy;
		delete[] sz;
		delete[] uxf;
		delete[] uyf;
		delete[] uzf;
		delete[] normg2f;
		delete[] x_p;
		delete[] y_p;
		delete[] z_p;
		delete[] normg2;
		delete[] jac;
		delete[] gx_x;
		delete[] gx_y;
		delete[] gx_z;
		delete[] gy_x;
		delete[] gy_y;
		delete[] gy_z;
		delete[] gz_x;
		delete[] gz_y;
		delete[] gz_z;
	
	}
}

void LogDemonsReg::setImages(float* F, float* M, unsigned int dimension[3]){
	src.fixed = F;
	src.moving = M;
	src.srcDim[0] = dimension[0];
	src.srcDim[1] = dimension[1];
	src.srcDim[2] = dimension[2];
	initialized = false;
}

void LogDemonsReg::initialize(){
	//initialize LogDemonsReg energy
	if (!initialized)
	{
		e_min = 1e10;

		//piggyback
		for (int i = 0; i < 3; ++i) {
			dim[i] = ceil((float)src.srcDim[i] * opt.pad);
			marginSize[i] = floor (src.srcDim[i]*(opt.pad-1.0f)/2.0f);
			len *= dim[i];
		}
		if (debug) 
			printf("Image Dimension: [0] - %i\t[1] - %i\t [2] - %i\nMargin Size: [0] - %i\t[1] - %i\t [2] - %i\nPadded Dimension: [0] - %i\t[1] - %i\t [2] - %i\n", src.srcDim[0], src.srcDim[1], src.srcDim[2], marginSize[0], marginSize[1], marginSize[2], dim[0], dim[1], dim[2]);
		printf("len: %i\n", len);

		fixed = new float[len]();
		moving = new float[len]();
		deformedMoving = new float[len]();

		// update field
		ux = new float[len]();
		uy = new float[len]();
		uz = new float[len]();

		// velocity field
		vx = new float[len]();
		vy = new float[len]();
		vz = new float[len]();

		// deformation field
		sx = new float[len]();
		sy = new float[len]();
		sz = new float[len]();

		// fixed image gradient
		uxf = new float[len]();
		uyf = new float[len]();
		uzf = new float[len]();
		normg2f = new float[len]();

		// temp memory space for computation
		x_p = new float[len]();
		y_p = new float[len]();
		z_p = new float[len]();
		normg2 = new float[len]();

		// memory for energy evaluation
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

		if (debug) printf("CPU Memory allocated.\n");
		padImage();
		LogDemonsReg::gradient(fixed, uxf, uyf, uzf, normg2f);

		initialized = true;
	}
}



void LogDemonsReg::padImage(){
	//Pad images
	for (unsigned int k = 0; k <src.srcDim[2]; ++k){
		for (unsigned int j = 0; j < src.srcDim[1]; ++j){

			// copy fixed and moving image with padding on a row-by-row basis
			memcpy(
				(fixed + (marginSize[2] + k)*(dim[0] * dim[1]) + (marginSize[1] + j)*(dim[0]) + marginSize[0]),
				(src.fixed + k*(src.srcDim[0] * src.srcDim[1]) + j*src.srcDim[0]),
				src.srcDim[0] * sizeof(float)
			);

			memcpy(
				(moving + (marginSize[2] + k)*(dim[0] * dim[1]) + (marginSize[1] + j)*(dim[0]) + marginSize[0]), 
				(src.moving + k*(src.srcDim[0] * src.srcDim[1]) + j*src.srcDim[0]),
				src.srcDim[0] * sizeof(float)
			);
		}
	}

	// At iteration == 0, Mp = M
	memcpy(deformedMoving, moving, len*sizeof(float));

	if (debug) printf("Image padded.\n");
}

void LogDemonsReg::Register()
{
	initialize();

	for (int iter = 0; iter < opt.iteration_max; ++iter) {

		/*	Given the current transformation s, compute a correspondence update field u
		by mimimizing E(u) w.r.t. u		*/

		this->findupdate();
		cout << "findupdate" << endl;


		//saveImage<float>(ux, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\ux.bin");
		//saveImage<float>(uy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\uy.bin");
		//saveImage<float>(uz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\uz.bin");

		/*	For a fluid-like regularization let u <- K(sigma_f) * u	*/
		this->imgaussian(ux, uy, uz, opt.sigma_f);
		cout << "imgaussian 0 " << endl;
	
		//saveImage<float>(ux, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\uxg.bin");
		//saveImage<float>(uy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\uyg.bin");
		//saveImage<float>(uz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\uzg.bin");

		/*	Let v <- v compose u	*/
	
		this->compose();
		cout << "compose " << endl;

		//saveImage<float>(vx, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\vx.bin");
		//saveImage<float>(vy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\vy.bin");
		//saveImage<float>(vz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\vz.bin");


		/*	For a diffusion like regularization let s <- K(sigma_d)*c (else, s<-c)	*/
		this->imgaussian(vx, vy, vz, opt.sigma_d);
		cout << "imgaussian 1 " << endl;
		//saveImage<float>(vx, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\vxg.bin");
		//saveImage<float>(vy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\vyg.bin");
		//saveImage<float>(vz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\vzg.bin");


		/*	s = exp(v)	*/
		this->expfield();	
		cout << "expfield " << endl;

		//saveImage<float>(sx, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\sx.bin");
		//saveImage<float>(sy, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\sy.bin");
		//saveImage<float>(sz, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\sz.bin");

		//Transform the moving image
		this->iminterpolate(moving, sx, sy, sz, deformedMoving);
		cout << "iminterpolate " << endl;

		//saveImage<float>(deformedMoving, dim, "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\iter1\\Mp.bin");

		//evulate energy
		energy_vec.push_back(energy());
		cout << "energy " << endl;
		printf("Iteration %i - Energy: %f\n", iter + 1, energy_vec.back());

		break;
		if (iter > 4){
			if ((energy_vec[iter - 5] - energy_vec[iter]) < (energy_vec[0] * opt.stop_criterium)){
				printf("e-5: %f\n", energy_vec[iter - 5]);
				printf("e: %f\n", energy_vec[iter]);
				printf("e-5 - e: %f\n", energy_vec[iter - 5] - energy_vec[iter]);
				printf("e[0] * opt.stop_criterium: %f\n", energy_vec[0] * opt.stop_criterium);
				break;
			}
		}
		std::string filename = "C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\cpu_results\\Mp_" + to_string(iter+1) + std::string(".bin");
		saveImage<float>(deformedMoving, dim, filename.c_str());
	}

	printf("LogDemonsReg Complete\n");	printf("\n"); printf("\n");
	for (int iter = 0; iter < energy_vec.size(); ++iter){
		printf("Iteration %i - Energy: %f\n", iter + 1, energy_vec[iter]);
	}
}

void LogDemonsReg::findupdate(){

	//Moving image gradient

	float alpha2 = (opt.sigma_i*opt.sigma_i) / (opt.sigma_x*opt.sigma_x);


	gradient(deformedMoving, ux, uy, uz, normg2);
	
	//Demons force formulation (current used: diffeo. mov.)
	//update is:
	//  Idiff / (||J||^2+(Idiff^2)/sigma_x^2) J
	//  with Idiff = F(x)-M(x+s), and J = Grad(M(x+s));

	//float* sgn_f = new float[len];
	for (unsigned int i = 0; i < len; i++) {
		float scale;
		float diff = fixed[i] - deformedMoving[i];

		int sgn = sign(ux[i] * uxf[i] + uy[i] * uyf[i] + uz[i] * uzf[i]);



		if (diff==0 || normg2[i] ==0) {
			scale = 0;
		} else {
			scale = diff / (normg2[i] + alpha2*diff*diff);
			(scale < 0) ? scale *= sgn : scale;
		}
		
		ux[i] *= scale;
		uy[i] *= scale;
		uz[i] *= scale;

	}

	//saveImage<float>(sgn_f, dim, "sgn.bin");
}

void LogDemonsReg::gradient(float *I, float *gx, float *gy, float *gz, float *norm)
{
	LogDemonsReg::gradient(I, gx, gy, gz);
	for (int i = 0; i < len; ++i){
		norm[i] = gx[i] * gx[i] + gy[i] * gy[i] + gz[i] * gz[i];
	}
}

void LogDemonsReg::gradient(float *I, float *gx, float *gy, float *gz)
{

	for (unsigned int k = 1; k < dim[2] - 1; ++k)
	{
		for (unsigned int j = 1; j < dim[1] - 1; ++j)
		{
			for (unsigned int i = 1; i < dim[0] - 1; ++i){
				int idx = pos(i, j, k);
				gx[idx] =
					(I[pos(i + 1, j, k)] - I[pos(i - 1, j, k)]) / 2.0f;

				gy[idx] =
					(I[pos(i, j + 1, k)] - I[pos(i, j - 1, k)]) / 2.0f;

				gz[idx] =
					(I[pos(i, j, k + 1)] - I[pos(i, j, k - 1)]) / 2.0f;

			}
		}
	}
}


// Composition of two fields : (u o v)
void LogDemonsReg::compose(){

	// add coordinate image
	for (unsigned int k = 0; k < dim[2]; ++k){
		for (unsigned int j = 0; j < dim[1]; ++j){
			for (unsigned int i = 0; i < dim[0]; ++i){
				int idx = pos(i, j, k);
				vx[idx] += i;
				vy[idx] += j;
				vz[idx] += k;
			}
		}
	}

	//// Original MATLAB code
	//iminterpolate(vx, ux, uy, uz, x_p);
	//iminterpolate(vy, ux, uy, uz, y_p);
	//iminterpolate(vz, ux, uy, uz, z_p);

	//memcpy(vx, x_p, sizeof(float)*len);
	//memcpy(vy, y_p, sizeof(float)*len);
	//memcpy(vz, z_p, sizeof(float)*len);

	//for (unsigned int k = 0; k < dim[2]; ++k){
	//	for (unsigned int j = 0; j < dim[1]; ++j){
	//		for (unsigned int i = 0; i < dim[0]; ++i){
	//			int idx = pos(i, j, k);
	//			vx[idx] -= i;
	//			vy[idx] -= j;
	//			vz[idx] -= k;
	//		}
	//	}
	//}

	// Rewritten code for performance
	for (unsigned int k = 0; k < dim[2]; ++k){
		for (unsigned int j = 0; j < dim[1]; ++j){
			for (unsigned int i = 0; i < dim[0]; ++i)
			{
				int idx = pos(i, j, k);

				//compute query point q which is pointed by the vector u
				float qx = i + ux[idx];
				float qy = j + uy[idx];
				float qz = k + uz[idx];

				if (qx < 0) qx = 0;
				if (qy < 0) qy = 0;
				if (qz < 0) qz = 0;
				if (qx > dim[0] - 1) qx = dim[0] - 1;
				if (qy > dim[1] - 1) qy = dim[1] - 1;
				if (qz > dim[2] - 1) qz = dim[2] - 1;

				//compute the vector vp by interpolation of field v at q
				interp3(vx, vy, vz, qx, qy, qz, &x_p[idx], &y_p[idx], &z_p[idx]);

				x_p[idx] -= i;
				y_p[idx] -= j;
				z_p[idx] -= k;
			}
		}
	}
	memcpy(vx, x_p, sizeof(float)*len);
	memcpy(vy, y_p, sizeof(float)*len);
	memcpy(vz, z_p, sizeof(float)*len);

	//clean up
	memset(x_p, 0, sizeof(float)*len);
	memset(y_p, 0, sizeof(float)*len);
	memset(z_p, 0, sizeof(float)*len);
}

//self_composition (s o s)
void LogDemonsReg::self_compose(){

	//coordinate image
	for (unsigned int k = 0; k < dim[2]; ++k){
		for (unsigned int j = 0; j < dim[1]; ++j){
			for (unsigned int i = 0; i < dim[0]; ++i){
				int idx = pos(i, j, k);
				sx[idx] += i;
				sy[idx] += j;
				sz[idx] += k;
			}
		}
	}


	for (unsigned int k = 0; k < dim[2]; ++k){
		for (unsigned int j = 0; j < dim[1]; ++j){
			for (unsigned int i = 0; i < dim[0]; ++i)
			{
				int idx = pos(i, j, k);

				//compute query point q which is pointed by the vector u
				float qx = sx[idx];
				float qy = sy[idx];
				float qz = sz[idx];

				if (qx < 0) qx = 0;
				if (qy < 0) qy = 0;
				if (qz < 0) qz = 0;
				if (qx > dim[0] - 1) qx = dim[0] - 1;
				if (qy > dim[1] - 1) qy = dim[1] - 1;
				if (qz > dim[2] - 1) qz = dim[2] - 1;
				//computer the vector vp by interpolation of field v at q
				interp3(sx, sy, sz, qx, qy, qz, &x_p[idx], &y_p[idx], &z_p[idx]);

				x_p[idx] -= i;
				y_p[idx] -= j;
				z_p[idx] -= k;
			}
		}
	}

	memcpy(sx, x_p, sizeof(float)*len);
	memcpy(sy, y_p, sizeof(float)*len);
	memcpy(sz, z_p, sizeof(float)*len);

	//clean up
	memset(x_p, 0, sizeof(float)*len);
	memset(y_p, 0, sizeof(float)*len);
	memset(z_p, 0, sizeof(float)*len);
}

void LogDemonsReg::iminterpolate(float* I, float* sx, float* sy, float* sz, float* Ip){
	
	for (unsigned int k = 0; k < dim[2]; ++k){
		for (unsigned int j = 0; j < dim[1]; ++j){
			for (unsigned int i = 0; i < dim[0]; ++i){
				int idx = pos(i, j, k);

				//compute query point q which is pointed by the vector u
				float qx = i + sx[idx];
				float qy = j + sy[idx];
				float qz = k + sz[idx];

				if (qx < 0) qx = 0;
				if (qy < 0) qy = 0;
				if (qz < 0) qz = 0;
				if (qx >= dim[0] - 1) qx = dim[0] - 1;
				if (qy >= dim[1] - 1) qy = dim[1] - 1;
				if (qz >= dim[2] - 1) qz = dim[2] - 1;

				//computer the vector vp by interpolation of field v at q
				//std::cout << idx << std::endl;;

				interp3(I, qx, qy, qz, &Ip[idx]);
			}
		}
	}
}

void LogDemonsReg::interp3(float* const ux, float* const uy, float* const uz,
	const float qx, const float qy, const float qz,
	float* const uxp, float* const uyp, float* const uzp){

	//Index of the closest grid point with the least coordinate
	int idx = pos(floor(qx), floor(qy), floor(qz));

	//Compute xd, yd, zd which are the differences between query point q 
	//	and the closest grid point with the least coordinate
	float xd = qx - floor(qx);
	float yd = qy - floor(qy);
	float zd = qz - floor(qz);

	//Variable declaration
	float wx[2][2][2][3];	//wx[k][j][i][dim]
	float wy[2][2][3];		//wy[k][j][dim]
	float wz[2][3];			//wz[k][dim]

	//Get 8 closest vectors {w000 ... w111} from q in field u for interpolation
	for (char k = 0; k < 2; ++k){
		for (char j = 0; j < 2; ++j){
			for (char i = 0; i < 2; ++i) {
				int xidx, yidx, zidx;
				if (i) xidx = ceil(qx); else xidx = floor(qx);
				if (j) yidx = ceil(qy); else yidx = floor(qy);
				if (k) zidx = ceil(qz); else zidx = floor(qz);

				int uidx = pos(xidx, yidx, zidx);
				wx[k][j][i][0] = ux[uidx];
				wx[k][j][i][1] = uy[uidx];
				wx[k][j][i][2] = uz[uidx];
			}
		}
	}

	//Interpolate {w000 ... w111} along x direction
	for (char k = 0; k < 2; ++k){
		for (char j = 0; j < 2; ++j){
			wy[k][j][0] = wx[k][j][0][0] * (1 - xd) + wx[k][j][1][0] * xd;
			wy[k][j][1] = wx[k][j][0][1] * (1 - xd) + wx[k][j][1][1] * xd;
			wy[k][j][2] = wx[k][j][0][2] * (1 - xd) + wx[k][j][1][2] * xd;
		}
	}

	//Interpolate {w00 ... w11} along y  direction
	for (char k = 0; k < 2; ++k){
		wz[k][0] = wy[k][0][0] * (1 - yd) + wy[k][1][0] * yd;
		wz[k][1] = wy[k][0][1] * (1 - yd) + wy[k][1][1] * yd;
		wz[k][2] = wy[k][0][2] * (1 - yd) + wy[k][1][2] * yd;
	}

	//Interpolate {w0 ... w1} along z direction
	*uxp = wz[0][0] * (1 - zd) + wz[1][0] * zd;
	*uyp = wz[0][1] * (1 - zd) + wz[1][1] * zd;
	*uzp = wz[0][2] * (1 - zd) + wz[1][2] * zd;
}

//Overloaded function for interpolation during image wrapping
void LogDemonsReg::interp3(float* const I, 	const float qx, const float qy, const float qz,
	float* const Ip){

	//Index of the closest grid point with the least coordinate
	int idx = pos(floor(qx), floor(qy), floor(qz));
	//Compute xd, yd, zd which are the differences between query point q 
	//	and the closest grid point with the least coordinate
	float xd = qx - floor(qx);
	float yd = qy - floor(qy);
	float zd = qz - floor(qz);

	//Variable declaration
	float Ix[2][2][2];	//wx[k][j][i]
	float Iy[2][2];		//wy[k][j]
	float Iz[2];		//wz[k]

	//Get 8 closest vectors {w000 ... w111} from q in field u for interpolation
	for (char k = 0; k < 2; ++k){
		for (char j = 0; j < 2; ++j){
			for (char i = 0; i < 2; ++i) {
				int xidx, yidx, zidx;
				i? xidx = ceil(qx): xidx = floor(qx);
				j? yidx = ceil(qy): yidx = floor(qy);
				k? zidx = ceil(qz): zidx = floor(qz);

				int uidx = pos(xidx, yidx, zidx);
				Ix[k][j][i] = I[uidx];
			}
		}
	}

	//Interpolate {w000 ... w111} along x direction
	for (char k = 0; k < 2; ++k){
		for (char j = 0; j < 2; ++j){
			Iy[k][j] = Ix[k][j][0] * (1 - xd) + Ix[k][j][1] * xd;
		}
	}

	//Interpolate {w00 ... w11} along y direction
	for (char k = 0; k < 2; ++k){
		Iz[k] = Iy[k][0] * (1 - yd) + Iy[k][1] * yd;
	}

	//Interpolate {w0 ... w1} along z direction
	*Ip = Iz[0] * (1 - zd) + Iz[1] * zd;
}


//field expontentials to get transformation
void LogDemonsReg::expfield() {

	//Choose N such that 2^-N is close to 0
	float mxnorm = 0;
	float normv2;
	for (unsigned int idx = 0; idx<len; ++idx){
		normv2 = vx[idx] * vx[idx] + vy[idx] * vy[idx] + vz[idx] * vz[idx];
		if (normv2 >mxnorm)  {
			mxnorm = normv2;
		}
	}

	cout << "largest normalized vector sqaured: " << mxnorm << endl;
	mxnorm = sqrt(mxnorm);
	int N = 0;
	while (mxnorm > 0.5f){
		N++;
		mxnorm *= 0.5;
	}

	//Perform explicit first order integration
	float scale = 1;
	for (int i = 0; i < N; i++) {
		scale /= 2;
	}
	
	for (unsigned int i = 0; i < len; ++i){
		sx[i] = vx[i] * scale;
		sy[i] = vy[i] * scale;
		sz[i] = vz[i] * scale;
	} 

	//Recursive scaling and squaring
	printf("self-composing for %i times...\n", N);
	for (int i = 0; i < N; ++i){
		self_compose();
	}
}

void LogDemonsReg::imgaussian(float* fx, float* fy, float* fz, float sigma) {
	int kernel_radius = int(ceil(3.0f * sigma));
	float* weight = new float[kernel_radius + 1];
	
	float sum = 0;
	for (int i = 0; i < kernel_radius + 1; ++i){
		weight[i] = exp(-(i*i) / (2 * sigma*sigma));
		i == 0 ? sum += weight[i] / 2.0: sum += weight[i];
	}
	for (int i = 0; i < kernel_radius + 1; ++i){
		weight[i] /= (2*sum);
	}
	
	//pass x
	for (unsigned int z = kernel_radius; z < dim[2] - kernel_radius; ++z){
		for (unsigned int y = kernel_radius; y < dim[1] - kernel_radius; ++y){
			for (unsigned int x = kernel_radius; x < dim[0] - kernel_radius; ++x){
				int idx = pos(x, y, z);

				for (int i = -kernel_radius; i <= kernel_radius; ++i){
					float this_weight = weight[abs(i)];
					x_p[idx] += this_weight*fx[pos(x + i, y, z)];
					y_p[idx] += this_weight*fy[pos(x + i, y, z)];
					z_p[idx] += this_weight*fz[pos(x + i, y, z)];
				}
			}
		}
	}
	memset(fx, 0, sizeof(float)*len);
	memset(fy, 0, sizeof(float)*len);
	memset(fz, 0, sizeof(float)*len);

	//pass y

	for (unsigned int z = kernel_radius; z < dim[2] - kernel_radius; ++z){
		for (unsigned int y = kernel_radius; y < dim[1] - kernel_radius; ++y){
			for (unsigned int x = kernel_radius; x < dim[0] - kernel_radius; ++x){
				int idx = pos(x, y, z);

				for (int i = -kernel_radius; i <= kernel_radius; ++i){
					float this_weight = weight[abs(i)];
					fx[idx] += this_weight*x_p[pos(x, y + i, z)];
					fy[idx] += this_weight*y_p[pos(x, y + i, z)];
					fz[idx] += this_weight*z_p[pos(x, y + i, z)];

				}
			}
		}
	}
	memset(x_p, 0, sizeof(float)*len);
	memset(y_p, 0, sizeof(float)*len);
	memset(z_p, 0, sizeof(float)*len);


	//pass z
	for (unsigned int z = kernel_radius; z < dim[2] - kernel_radius; ++z){
		for (unsigned int y = kernel_radius; y < dim[1] - kernel_radius; ++y){
			for (unsigned int x = kernel_radius; x < dim[0] - kernel_radius; ++x){
				int idx = pos(x, y, z);

				for (int i = -kernel_radius; i <= kernel_radius; ++i){
					float this_weight = weight[abs(i)];
					x_p[idx] += this_weight*fx[pos(x, y, z + i)];
					y_p[idx] += this_weight*fy[pos(x, y, z + i)];
					z_p[idx] += this_weight*fz[pos(x, y, z + i)];

				}
			}
		}
	}

	memcpy(fx, x_p, sizeof(float)*len);
	memcpy(fy, y_p, sizeof(float)*len);
	memcpy(fz, z_p, sizeof(float)*len);


	//clean up
	memset(x_p, 0, sizeof(float)*len);
	memset(y_p, 0, sizeof(float)*len);
	memset(z_p, 0, sizeof(float)*len);
	delete[] weight;
	


}


void LogDemonsReg::jacobian(float* det_J){
	gradient(sx, gx_x, gx_y, gx_z);
	gradient(sy, gy_x, gy_y, gy_z);
	gradient(sz, gz_x, gz_y, gz_z);

	for (unsigned int i = 0; i < len; ++i){
		gx_x[i] = gx_x[i] + 1;
		gy_y[i] = gy_y[i] + 1;
		gz_z[i] = gz_z[i] + 1;

		det_J[i] = 
			gx_x[i] * gy_y[i] * gz_z[i] +
			gy_x[i] * gz_y[i] * gx_z[i] +
			gz_x[i] * gx_y[i] * gy_z[i] -
			gz_x[i] * gy_y[i] * gx_z[i] -
			gy_x[i] * gx_y[i] * gz_z[i] -
			gx_x[i] * gz_y[i] * gy_z[i];

		det_J[i] = det_J[i] * det_J[i];
	}

	// clean up
	memset(gx_x, 0, sizeof(float)*len);
	memset(gx_y, 0, sizeof(float)*len);
	memset(gx_z, 0, sizeof(float)*len);
	memset(gy_x, 0, sizeof(float)*len);
	memset(gy_y, 0, sizeof(float)*len);
	memset(gy_z, 0, sizeof(float)*len);
	memset(gz_x, 0, sizeof(float)*len);
	memset(gz_y, 0, sizeof(float)*len);
	memset(gz_z, 0, sizeof(float)*len);
}

float LogDemonsReg::energy(){

	jacobian(jac);

	float e = 0;
	float diff2n;
	float reg_weight = (opt.sigma_i*opt.sigma_i) / (opt.sigma_x*opt.sigma_x);
	cout <<"reg_weight = "<< reg_weight << endl;
	for (unsigned int i = 0; i < len; ++i){
		diff2n = (fixed[i] - deformedMoving[i])*(fixed[i] - deformedMoving[i]);

		e += (diff2n + reg_weight *jac[i])/float(len);
	}

	// clean up
	//memset(jac, 0, sizeof(float)*len);

	return e;
}

