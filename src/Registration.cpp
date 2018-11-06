#include "Registration.h"



//Public Functions

//Default constructor

Registration::Registration(float *fixed, float *moving, unsigned int dimension[3])
{
	int len = dimension[0] * dimension[1] * dimension[2];
	fixedsrc = new SourcePtr(fixed);
	movingsrc = new SourcePtr(moving);
	memcpy(dim, dimension, 3 * sizeof(unsigned int));
	loaded = true;
}

Registration::~Registration()
{
	if (loaded){
		delete fixedsrc;
		delete movingsrc;
	}
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
	}
}

void Registration::debugOn(bool dbg){
	debug = dbg;
}

void Registration::setImages(float& F, float& M, unsigned int dimension[3]){
	int len = dim[0] * dim[1] * dim[2];


	fixedsrc = new SourcePtr(&F);
	movingsrc = new SourcePtr(&M);
	memcpy(dim, dimension, 3 * sizeof(unsigned int));
	loaded = true;
}

void Registration::initialize(){
	//initialize registration energy
	if (!initialized)
	{
		e = new float[opt.iteration_max]();
		e_min = 1e10;

		//piggyback
		for (int i = 0; i < 3; ++i) {
			marginSize[i] = ((dim[i] * opt.pad) - dim[i]) / 2;
			paddedDim[i] = marginSize[i] * 2 + dim[i];
			paddedlen *= paddedDim[i];
		}
		if (debug) printf("Margin Size: [0] - %i\t[1] - %i\t [2] - %i\nPadded Dimension: [0] - %i\t[1] - %i\t [2] - %i\n",
			marginSize[0], marginSize[1], marginSize[2], paddedDim[0], paddedDim[1], paddedDim[2]);

		fixed = new float[paddedlen]();
		moving = new float[paddedlen]();
		deformedMoving = new float[paddedlen]();
		ux = new float[paddedlen]();
		uy = new float[paddedlen]();
		uz = new float[paddedlen]();
		vx = new float[paddedlen]();
		vy = new float[paddedlen]();
		vz = new float[paddedlen]();
		sx = new float[paddedlen]();
		sy = new float[paddedlen]();
		sz = new float[paddedlen]();

		if (debug) printf("CPU Memory allocated.\n");
		padImage();
		initialized = true;
	}
}

void Registration::padImage(){
	//Pad images
	for (unsigned int k = 0; k <dim[2]; ++k){
		for (unsigned int j = 0; j < dim[1]; ++j){
			//copy fixed image on row-by-row basis
			memcpy((fixed + (marginSize[2] + k)*(paddedDim[0] * paddedDim[1]) + (marginSize[1] + j)*(paddedDim[0]) + marginSize[0]),
				(fixedsrc->data + k*(dim[0] * dim[1]) + j*dim[0]),
				dim[0] * sizeof(float));

			//copy moving image on row-by-row basis to moving and deformedMoving
			memcpy((moving + (marginSize[2] + k)*(paddedDim[0] * paddedDim[1]) + (marginSize[1] + j)*(paddedDim[0]) + marginSize[0]),
				(movingsrc->data + k*(dim[0] * dim[1]) + j*dim[0]),
				dim[0] * sizeof(float));

			memcpy((deformedMoving + (marginSize[2] + k)*(paddedDim[0] * paddedDim[1]) + (marginSize[1] + j)*(paddedDim[0]) + marginSize[0]),
				(movingsrc->data + k*(dim[0] * dim[1]) + j*dim[0]),
				dim[0] * sizeof(float));
		}
	}
	if (debug) printf("Image padded.\n");
}

void Registration::Register()
{
	char* c_it = new char;
	initialize();
	int iter = 0;

	high_resolution_clock::time_point t1;
	high_resolution_clock::time_point t2;



	for (int iter = 0; iter < opt.iteration_max; ++iter) {

		/*	Given the current transformation s, compute a correspondence update field u
		by mimimizing E(u) w.r.t. u		*/
		t1 = high_resolution_clock::now();
		
		this->findupdate(fixed, deformedMoving, ux, uy, uz);
		
		
		t2 = high_resolution_clock::now();
		auto duration = duration_cast<milliseconds>(t2 - t1);
		cout << "time required for update" << duration.count() << " ms" << endl;


		/*	For a fluid-like regularization let u <- K(sigma_f) * u	*/
		t1 = high_resolution_clock::now();

		this->imgaussian(ux, uy, uz, opt.sigma_f);

		t2 = high_resolution_clock::now();
		 duration = duration_cast<milliseconds>(t2 - t1);
		cout << "time required for gaussian" << duration.count() << " ms" << endl;

		/*	Let v <- v compose u	*/

		t1 = high_resolution_clock::now();

		this->compose(vx, vy, vz, ux, uy, uz);

		t2 = high_resolution_clock::now();
		 duration = duration_cast<milliseconds>(t2 - t1);
		cout << "time required for compose" << duration.count() << " ms" << endl;

		/*	For a diffusion like regularization let s <- K(sigma_d)*c (else, s<-c)	*/
		this->imgaussian(vx, vy, vz, opt.sigma_d);

		/*	s = exp(v)	*/
		this->expfield(vx, vy, vz, sx, sy, sz);	

		//Transform the moving image

		t1 = high_resolution_clock::now();
		this->iminterpolate(moving, sx, sy, sz, deformedMoving);


		t2 = high_resolution_clock::now();
		 duration = duration_cast<milliseconds>(t2 - t1);
		cout << "time required for interpolation" << duration.count() << " ms" << endl;


		//evulate energy

		t1 = high_resolution_clock::now();
		e[iter] = energy(fixed, deformedMoving, sx, sy, sz);


		t2 = high_resolution_clock::now();
		 duration = duration_cast<milliseconds>(t2 - t1);
		cout << "time required for energy" << duration.count() << " ms" << endl;


		printf("Iteration %i - Energy: %f\n", iter + 1, e[iter]);

		if (iter > 4){
			if ((e[iter - 5] - e[iter]) < (e[0] * opt.stop_criterium)){
				printf("e-5: %f\n", e[iter - 5]);
				printf("e: %f\n", e[iter]);
				printf("e-5 - e: %f\n", e[iter - 5] - e[iter]);
				printf("e[0] * opt.stop_criterium: %f\n", e[0] * opt.stop_criterium);
				break;
			}
		}
	}

	printf("Registration Complete\n");	printf("\n"); printf("\n");
	for (int iter = 0; iter < opt.iteration_max; ++iter){
		printf("Iteration %i - Energy: %f\n", iter + 1, e[iter]);
	}

	saveResult(deformedMoving, sx, sy, sz);
}

void Registration::saveResult(float* Mp, float* sx, float* sy, float* sz){
	stringstream filename;

	filename.str("");
	filename << opt.output_header << "Mp" << ".bin";
	saveSlice<float>(deformedMoving, paddedDim, filename.str());

	filename.str("");
	filename << opt.output_header << "sx" << ".bin";
	saveSlice<float>(sx, paddedDim, filename.str());

	filename.str("");
	filename << opt.output_header << "sy" << ".bin";
	saveSlice<float>(sy, paddedDim, filename.str());

	filename.str("");
	filename << opt.output_header << "sz" << ".bin";
	saveSlice<float>(sz, paddedDim, filename.str());
}


//Private function
template <typename T> inline int Registration::sign(T val) {
	return (T(0) < val) - (val < T(0));
}

//Get array index from position
inline unsigned int Registration::pos(int i, int j, int k){
	return  i + j*paddedDim[0] + k*paddedDim[0] * paddedDim[1];
}


void Registration::findupdate(float* F, float* Mp, float* ux, float* uy, float* uz){

	//Moving image gradient

	float alpha2 = (opt.sigma_i*opt.sigma_i) / (opt.sigma_x*opt.sigma_x);

	float *normg2 = new float[paddedlen]();
	gradient(Mp, ux, uy, uz, normg2);

	float *uxf = new float[paddedlen]();
	float *uyf = new float[paddedlen]();
	float *uzf = new float[paddedlen]();
	float *normg2f = new float[paddedlen]();
	gradient(F, uxf, uyf, uzf, normg2f);

	//Demons force formulation (current used: diffeo. mov.)
	//update is:
	//  Idiff / (||J||^2+(Idiff^2)/sigma_x^2) J
	//  with Idiff = F(x)-M(x+s), and J = Grad(M(x+s));

	
	for (unsigned int i = 0; i < paddedlen; i++) {
		float scale;
		float diff = F[i] - Mp[i];


		int sgn = sign(ux[i] * uxf[i] + uy[i] * uyf[i] + uz[i] * uzf[i]);

		if (diff ==0 || normg2[i] ==0) {
			scale = 0;
		} else {
			scale = diff / (normg2[i] + alpha2*diff*diff);
			(scale < 0) ? scale *= sgn : scale;
		}
		
		ux[i] *= scale;
		uy[i] *= scale;
		uz[i] *= scale;
	}

	delete[] normg2;
	delete[] uxf;
	delete[] uyf;
	delete[] uzf;
	delete[] normg2f;

}

void Registration::gradient(float *I, float *gx, float *gy, float *gz, float *normg2)
{

	for (unsigned int k = 1; k < paddedDim[2] - 1; ++k)
	{
		for (unsigned int j = 1; j < paddedDim[1] - 1; ++j)
		{
			for (unsigned int i = 1; i < paddedDim[0] - 1; ++i){
				int idx = pos(i, j, k);
				gx[idx] =
					(I[pos(i + 1, j, k)] - I[pos(i - 1, j, k)]) / 2;

				gy[idx] =
					(I[pos(i, j + 1, k)] - I[pos(i, j - 1, k)]) / 2;


				gz[idx] =
					(I[pos(i, j, k + 1)] - I[pos(i, j, k - 1)]) / 2;

				normg2[idx] =
					gx[idx] *gx[idx] +
					gy[idx] *gy[idx] +
					gz[idx] *gz[idx];
			}
		}
	}
}

void Registration::gradient(float *I, float *gx, float *gy, float *gz)
{

	for (unsigned int k = 1; k < paddedDim[2] - 1; ++k)
	{
		for (unsigned int j = 1; j < paddedDim[1] - 1; ++j)
		{
			for (unsigned int i = 1; i < paddedDim[0] - 1; ++i){
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


// Composition of two fields : (U o V)
void Registration::compose(float* vx, float* vy, float* vz, float* ux, float* uy, float* uz){

	float *x_p, *y_p, *z_p;

	//create tmp data variable for self-composing
	x_p = new float[paddedlen]();
	y_p = new float[paddedlen]();
	z_p = new float[paddedlen]();

	//coordinate image
	for (unsigned int k = 0; k < paddedDim[2]; ++k){
		for (unsigned int j = 0; j < paddedDim[1]; ++j){
			for (unsigned int i = 0; i < paddedDim[0]; ++i){
				int idx = pos(i, j, k);
				vx[idx] += i;
				vy[idx] += j;
				vz[idx] += k;
			}
		}
	}


	for (unsigned int k = 0; k < paddedDim[2]; ++k){
		for (unsigned int j = 0; j < paddedDim[1]; ++j){
			for (unsigned int i = 0; i < paddedDim[0]; ++i)
			{
				int idx = pos(i, j, k);

				//compute query point q which is pointed by the vector u
				float qx = i + ux[idx];
				float qy = j + uy[idx];
				float qz = k + uz[idx];

				if (qx < 0) qx = 0;
				if (qy < 0) qy = 0;
				if (qz < 0) qz = 0;
				if (qx > paddedDim[0]-1) qx = paddedDim[0]-1;
				if (qy > paddedDim[1]-1) qy = paddedDim[1]-1;
				if (qz > paddedDim[2]-1) qz = paddedDim[2]-1;
				//computer the vector vp by interpolation of field v at q
				interp3(vx, vy, vz, qx, qy, qz, &x_p[idx], &y_p[idx], &z_p[idx]);

				x_p[idx] -= i;
				y_p[idx] -= j;
				z_p[idx] -= k;
			}
		}
	}
	memcpy(vx, x_p, sizeof(float)*paddedlen);
	memcpy(vy, y_p, sizeof(float)*paddedlen);
	memcpy(vz, z_p, sizeof(float)*paddedlen);

	//clean up
	delete[] x_p;
	delete[] y_p;
	delete[] z_p;
}

//self_composition (U o U)
void Registration::compose(float* vx, float* vy, float* vz){

	float *x_p, *y_p, *z_p;

	//create tmp data variable for self-composing
	x_p = new float[paddedlen]();
	y_p = new float[paddedlen]();
	z_p = new float[paddedlen]();

	//coordinate image
	for (unsigned int k = 0; k < paddedDim[2]; ++k){
		for (unsigned int j = 0; j < paddedDim[1]; ++j){
			for (unsigned int i = 0; i < paddedDim[0]; ++i){
				int idx = pos(i, j, k);
				vx[idx] += i;
				vy[idx] += j;
				vz[idx] += k;
			}
		}
	}


	for (unsigned int k = 0; k < paddedDim[2]; ++k){
		for (unsigned int j = 0; j < paddedDim[1]; ++j){
			for (unsigned int i = 0; i < paddedDim[0]; ++i)
			{
				int idx = pos(i, j, k);

				//compute query point q which is pointed by the vector u
				float qx = vx[idx];
				float qy = vy[idx];
				float qz = vz[idx];

				if (qx < 0) qx = 0;
				if (qy < 0) qy = 0;
				if (qz < 0) qz = 0;
				if (qx > paddedDim[0] - 1) qx = paddedDim[0] - 1;
				if (qy > paddedDim[1] - 1) qy = paddedDim[1] - 1;
				if (qz > paddedDim[2] - 1) qz = paddedDim[2] - 1;
				//computer the vector vp by interpolation of field v at q
				interp3(vx, vy, vz, qx, qy, qz, &x_p[idx], &y_p[idx], &z_p[idx]);

				x_p[idx] -= i;
				y_p[idx] -= j;
				z_p[idx] -= k;
			}
		}
	}
	memcpy(vx, x_p, sizeof(float)*paddedlen);
	memcpy(vy, y_p, sizeof(float)*paddedlen);
	memcpy(vz, z_p, sizeof(float)*paddedlen);

	//clean up
	delete[] x_p;
	delete[] y_p;
	delete[] z_p;
}


void Registration::interp3(float* const ux, float* const uy, float* const uz,
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
void Registration::interp3(float* const I, 	const float qx, const float qy, const float qz,
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

void Registration::iminterpolate(float* I, float* sx, float* sy, float* sz, float* Ip){
	//TODO Re-implement the iminterpolate function
	for (unsigned int k = 0; k < paddedDim[2]; ++k){
		for (unsigned int j = 0; j < paddedDim[1]; ++j){
			for (unsigned int i = 0; i < paddedDim[0]; ++i){
				int idx = pos(i, j, k);

				//compute query point q which is pointed by the vector u
				float qx = i + sx[idx];
				float qy = j + sy[idx];
				float qz = k + sz[idx];

				if (qx < 0) qx = 0;
				if (qy < 0) qy = 0;
				if (qz < 0) qz = 0;
				if (qx >= paddedDim[0] - 1) qx = paddedDim[0] - 1;
				if (qy >= paddedDim[1] - 1) qy = paddedDim[1] - 1;
				if (qz >= paddedDim[2] - 1) qz = paddedDim[2] - 1;

				//computer the vector vp by interpolation of field v at q
				//std::cout << idx << std::endl;;
				
				interp3(I, qx, qy, qz, &Ip[idx]);
			}
		}
	}

}

//field expontentials to get transformation
void Registration::expfield(float* vx, float* vy, float* vz, float* sx, float* sy, float* sz) {

	//Choose N such that 2^-N is close to 0
	float mxnorm = 0;
	float normv2;
	for (unsigned int k = 0; k < paddedDim[2]; ++k){
		for (unsigned int j = 0; j < paddedDim[1]; ++j){
			for (unsigned int i = 0; i < paddedDim[0]; ++i){
				int idx = pos(i, j, k);
				normv2 = vx[idx] * vx[idx] + vy[idx] * vy[idx] + vz[idx] * vz[idx];
				if (normv2 > mxnorm) {
					mxnorm = normv2;
				}
			}
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
	float scale = pow((float)2, -N);
	for (unsigned int i = 0; i < paddedlen; ++i){
		sx[i] = vx[i] * scale;
		sy[i] = vy[i] * scale;
		sz[i] = vz[i] * scale;
	} 

	//Recursive scaling and squaring
	printf("self-composing for %i times...\n", N);
	for (int i = 0; i < N; ++i){
		compose(sx, sy, sz);
	}
}







void Registration::imgaussian(float* ux, float* uy, float* uz, float sigma) {
	int kernel_radius = int(3.0f * sigma);
	float* weight = new float[kernel_radius + 1];

	for (int i = 0; i < kernel_radius + 1; ++i){
		weight[i]=(0.39894228f / sigma) * exp(-(i*i) / (2 * sigma*sigma));
	}

	float* ux_t = new float[paddedlen]();
	float* uy_t = new float[paddedlen]();
	float* uz_t = new float[paddedlen]();

	//pass x
	for (unsigned int z = kernel_radius; z < dim[2] - kernel_radius; ++z){
		for (unsigned int y = kernel_radius; y < dim[1] - kernel_radius; ++y){
			for (unsigned int x = kernel_radius; x < dim[0] - kernel_radius; ++x){
				int idx = pos(x, y, z);

				for (int i = -kernel_radius; i <= kernel_radius; ++i){
					float this_weight = weight[abs(i)];
					ux_t[idx] += this_weight*ux[pos(x + i, y, z)];
					uy_t[idx] += this_weight*uy[pos(x + i, y, z)];
					uz_t[idx] += this_weight*uz[pos(x + i, y, z)];

				}
			}
		}
	}
	memset(ux, 0, sizeof(float)*paddedlen);
	memset(uy, 0, sizeof(float)*paddedlen);
	memset(uz, 0, sizeof(float)*paddedlen);

	//pass y

	for (unsigned int z = kernel_radius; z < dim[2] - kernel_radius; ++z){
		for (unsigned int y = kernel_radius; y < dim[1] - kernel_radius; ++y){
			for (unsigned int x = kernel_radius; x < dim[0] - kernel_radius; ++x){
				int idx = pos(x, y, z);

				for (int i = -kernel_radius; i <= kernel_radius; ++i){
					float this_weight = weight[abs(i)];
					ux[idx] += this_weight*ux_t[pos(x, y + i, z)];
					uy[idx] += this_weight*uy_t[pos(x, y + i, z)];
					uz[idx] += this_weight*uz_t[pos(x, y + i, z)];

				}
			}
		}
	}
	memset(ux_t, 0, sizeof(float)*paddedlen);
	memset(uy_t, 0, sizeof(float)*paddedlen);
	memset(uz_t, 0, sizeof(float)*paddedlen);


	//pass z
	for (unsigned int z = kernel_radius; z < dim[2] - kernel_radius; ++z){
		for (unsigned int y = kernel_radius; y < dim[1] - kernel_radius; ++y){
			for (unsigned int x = kernel_radius; x < dim[0] - kernel_radius; ++x){
				int idx = pos(x, y, z);

				for (int i = -kernel_radius; i <= kernel_radius; ++i){
					float this_weight = weight[abs(i)];
					ux_t[idx] += this_weight*ux[pos(x, y, z + i)];
					uy_t[idx] += this_weight*uy[pos(x, y, z + i)];
					uz_t[idx] += this_weight*uz[pos(x, y, z + i)];

				}
			}
		}
	}
	memcpy(ux, ux_t, sizeof(float)*paddedlen);
	memcpy(uy, uy_t, sizeof(float)*paddedlen);
	memcpy(uz, uz_t, sizeof(float)*paddedlen);

	delete[] ux_t;	
	delete[] uy_t;
	delete[] uz_t;
	delete[] weight;
	


}


void Registration::jacobian(float* det_J, float* sx, float* sy, float* sz){
	float* gx_x = new float[paddedlen]();
	float* gx_y = new float[paddedlen]();
	float* gx_z = new float[paddedlen]();
	float* gy_x = new float[paddedlen]();
	float* gy_y = new float[paddedlen]();
	float* gy_z = new float[paddedlen]();
	float* gz_x = new float[paddedlen]();
	float* gz_y = new float[paddedlen]();
	float* gz_z = new float[paddedlen]();

	gradient(sx, gx_x, gx_y, gx_z);
	gradient(sy, gy_x, gy_y, gy_z);
	gradient(sz, gz_x, gz_y, gz_z);

	for (int i = 0; i < paddedlen; ++i){
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
	}
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

float Registration::energy(float* F, float* Mp, float* sx, float* sy, float* sz){

	float* jac = new float[paddedlen]();
	jacobian(jac, sx, sy, sz);

	float e = 0;
	float diff2n;
	float reg_weight = (opt.sigma_i*opt.sigma_i) / (opt.sigma_x*opt.sigma_x);
	cout <<"reg_weight = "<< reg_weight << endl;
	for (int i = 0; i < paddedlen; ++i){
		diff2n = (F[i] - Mp[i])*(F[i] - Mp[i]) / (float)paddedlen;
		jac[i] = jac[i] * jac[i] / (float)paddedlen;
		e += diff2n + reg_weight *jac[i];
	}

	return e;
}

