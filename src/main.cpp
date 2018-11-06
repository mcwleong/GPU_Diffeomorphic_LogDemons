#define ECHO
//#define TEST

#define CPU 1


//Std lib include
#include <stdio.h>
#include <cstdlib>
#include <fstream>

//GDCM includes
#include "gdcmStringFilter.h"
#include "gdcmAttribute.h"
#include "gdcmImage.h"
#include "gdcmImageReader.h"
#include "gdcmImageWriter.h"
#include "gdcmException.h"


//Custom include
#include "Registration.h"
#include "RegistrationGPU.cuh"

using namespace gdcm;

void castToFloat(Pixmap *pixmap, char* Src, float* Dst, int nPix){


	if (pixmap->GetPixelFormat() == PixelFormat::INT16){
		unsigned short* sdata = (unsigned short*)Src;
		unsigned short largest = 0;
		for (int i = 0; i < nPix; ++i){
			Dst[i] = (float)sdata[i];
		}
	}
	else if (pixmap->GetPixelFormat() == PixelFormat::UINT8){
		unsigned char* uSrc = (unsigned char*)Src;
		for (int i = 0; i < nPix; ++i){
			Dst[i] = (float)uSrc[i];
		}
	}
	else {
		cerr << "Unsupported pixel formet. Supports UINT8 and INT16.\n"
			<< "Input pixel type is: \n" << 
			pixmap->GetPixelFormat();
		getchar(); exit(1);
	}
}


int main(int argc, char* argv[])
{
	//Real Meat
	
	/*
	//fuck the dicom readers
	if (argc < 3) {
		std::cerr << "Usage: cudaDiffeomorphic [fixedImage] [movingImage]";
		getchar();
		return 1;
	}
	
	ImageReader fixedimgReader, movingimgReader;
	char* fixedFilename = argv[1];
	char* movingFilename = argv[2];
	printf("Loading file: %s\n", fixedFilename);
	fixedimgReader.SetFileName(fixedFilename);		//FixedBrain.dcm
	printf("Loading file: %s\n", movingFilename);
	movingimgReader.SetFileName(movingFilename);	//MovingBrain.dcm


	//read Dicom File
	if (!fixedimgReader.Read()) {
		std::cerr << "Failed to read fixed Image, check file validity.";
		getchar();		return 1;
	}
	if (!movingimgReader.Read()) {
		std::cerr << "Failed to read moving Image, check file validity.";
		getchar();		return 1;
	}
	
	Pixmap fixedPixmap = fixedimgReader.GetPixmap();
	Pixmap movingPixmap = movingimgReader.GetPixmap();
	
	if ((fixedPixmap.GetNumberOfDimensions() != 3) && 
		(movingPixmap.GetNumberOfDimensions() != 3)){
		std::cerr << "The input files have to be a 3D image.";
		getchar(); 		return 1;
	}
	int nPix = 1;
	unsigned int dim[3];
	for (int i = 0; i < 3; ++i){
		if ((fixedPixmap.GetDimension(i) != movingPixmap.GetDimension(i))){
			std::cerr << "The input files have to be at the same dimension.";
			getchar(); 		return 1;
		}
		else{
			dim[i] = fixedPixmap.GetDimension(i);
			nPix *= dim[i];
		}
	}

#ifdef ECHO
	printf("Number of Pixels: %i\n", nPix);
	printf("Dimensions: ");
	for (int i = 0; i < 3; ++i){
		if (i > 0) printf(" x ");
		printf("%i", dim[i]);
	}
	printf("\n");
	std::cout << fixedPixmap.GetPixelFormat() << std::endl;
	std::cout << movingPixmap.GetPixelFormat() << std::endl;

#endif

	//Load data from image buffer
	char* fixedRawData = (char*)malloc(fixedPixmap.GetBufferLength()*sizeof(unsigned char));
	if (!fixedPixmap.GetBuffer(fixedRawData)) {
		std::cerr << "Cannot load fixed data from image buffer.";
		getchar();	return 1;
	}

	char* movingRawData = (char*)malloc(movingPixmap.GetBufferLength()*sizeof(unsigned char));
	if (!movingPixmap.GetBuffer(movingRawData)) {
		std::cerr << "Cannot load moving data from image buffer.";
		getchar();	return 1;
	}


	*/

	//cast the data to float
	
	float *fixedData, *movingData;
	unsigned int dim[3];
	const char* FixPath = "H:\\diffeomorphic_cuda\\data\\GeneratedImageSets\\FixedBrain_1.8.bin";
	const char* MovPath = "H:\\diffeomorphic_cuda\\data\\GeneratedImageSets\\MovingBrain_1.8.bin";

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


	{
		float largest = 0;
		for (int i = 0; i < len; ++i){
			float t;
			fixedData[i] > movingData[i] ? t = fixedData[i] : t = movingData[i];
			if (t > largest) largest = t ;
		}
		for (int i = 0; i < len; ++i){
			fixedData[i] = (fixedData[i] / largest) * 255;
			movingData[i] = (movingData[i] / largest) * 255;
		}

	}


	//castToFloat(&fixedPixmap, fixedRawData, fixedData, nPix);
	//castToFloat(&movingPixmap, movingRawData, movingData, nPix);

	//delete[] fixedRawData, movingRawData;

	RegistrationGPU reg(fixedData, movingData, dim);
	reg.debugOn(false);
	reg.opt.iteration_max = 1;
	reg.opt.stop_criterium = 0.025f;
	//reg.Register();
	reg.Registration::Register();

	//Cleanup
	delete[] fixedData, movingData;

	std::cout << "Program exits, Press enter...";
	
	getchar();
	return 0;
}
