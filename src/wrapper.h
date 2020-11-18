#pragma once
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include <iostream>

//change input pixel type if needed
typedef unsigned int InputPixelType;
typedef float OutputPixelType;

typedef itk::Image<unsigned int, 3> ImageType;
typedef itk::Image<float, 3> OutputImageType;

typedef itk::ImageFileReader<itk::Image<unsigned int, 3>> ReaderType;
typedef itk::CastImageFilter<itk::Image<unsigned int, 3>, itk::Image<float, 3>> CastImageFilterType;
typedef itk::RescaleIntensityImageFilter<itk::Image<unsigned int, 3>, itk::Image<unsigned int, 3>> RescaleFilterType;
typedef itk::Size<3> SizeType;

ImageType::Pointer readDicomImg(const char* filename) {
	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName(filename);
	reader->Update();
	return reader->GetOutput();
}

ImageType::Pointer rescaleImage(ImageType::Pointer img) {

	RescaleFilterType::Pointer rescale = RescaleFilterType::New();
	rescale->SetOutputMinimum(0);
	rescale->SetOutputMaximum(255);
	rescale->SetInput(img);
	rescale->Update();

	return rescale->GetOutput();
}

OutputImageType::Pointer castImage(ImageType::Pointer img) {

	CastImageFilterType::Pointer castFilter = CastImageFilterType::New();
	castFilter->SetInput(img);
	castFilter->Update();
	return castFilter->GetOutput();

}

void saveImageToBuffer(OutputImageType::Pointer img, float*& ptr, unsigned int(&dim)[3]) {
	SizeType size = img->GetLargestPossibleRegion().GetSize();
	dim[0] = size[0]; dim[1] = size[1]; dim[2] = size[2];
	//std::cout << size << std::endl;
	unsigned int len = 1;
	for (int i = 0; i < 3; ++i) {
		len *= dim[i];
	}

	ptr = new float[len];
	memcpy(ptr, img->GetBufferPointer(), dim[0] * dim[1] * dim[2] * sizeof(float));
}

void readDicomToBuffer(const char* filename, float*& dataptr, unsigned int(&dim)[3]) {
	ImageType::Pointer img = readDicomImg(filename);
	img = rescaleImage(img);
	OutputImageType::Pointer floatimg = castImage(img);
	saveImageToBuffer(floatimg, dataptr, dim);

}