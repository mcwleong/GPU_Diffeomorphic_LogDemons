#pragma once
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkImportImageFilter.h"
#include <iostream>

//change input pixel type if needed
typedef unsigned int InputPixelType;
typedef float OutputPixelType;

typedef itk::Image<unsigned short, 3> ImageType;
typedef itk::Image<float, 3> OutputImageType;
typedef itk::Size<3> SizeType;

ImageType::Pointer readDicomImg(const char* filename) {
	typedef itk::ImageFileReader<ImageType> ReaderType;

	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName(filename);
	reader->Update();
	return reader->GetOutput();
}

void writefloatResultToDicomImage(OutputImageType::Pointer img, char* filename) {
	typedef itk::CastImageFilter<OutputImageType, ImageType> CastImageFilterType;
	CastImageFilterType::Pointer castFilter = CastImageFilterType::New();
	castFilter->SetInput(img);
		
	try
	{
		castFilter->Update();
	}
	catch (itk::ExceptionObject& error)
	{
		std::cerr << "Error: " << error << std::endl;
	}	
	
	typedef itk::ImageFileWriter<ImageType> WriterType;
	ImageType::Pointer casted = castFilter->GetOutput();
	//std::cout<<casted->GetLargestPossibleRegion().GetSize();
		
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(filename);
	writer->SetInput(casted);

	try
	{
		writer->Update();
	}
	catch (itk::ExceptionObject& error)
	{
		std::cerr << "Error: " << error << std::endl;
	}	
}


ImageType::Pointer rescaleImage(ImageType::Pointer img) {
	typedef itk::RescaleIntensityImageFilter<ImageType, ImageType> RescaleFilterType;

	RescaleFilterType::Pointer rescale = RescaleFilterType::New();
	rescale->SetOutputMinimum(0);
	rescale->SetOutputMaximum(255); //amending the rescale intensity value may increase registration efficiency
	rescale->SetInput(img);
	rescale->Update();

	return rescale->GetOutput();
}

OutputImageType::Pointer castImage(ImageType::Pointer img) {
	typedef itk::CastImageFilter<ImageType, OutputImageType> CastImageFilterType;

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

OutputImageType::Pointer readBufferToITKImagePtr(float* buffer, unsigned int (&dim)[3]) {

	typedef itk::ImportImageFilter<float, 3> ImportFilterType;
	ImportFilterType::Pointer importFilter = ImportFilterType::New();
	ImportFilterType::SizeType size;
	size[0] = dim[0];
	size[1] = dim[1];
	size[2] = dim[2];

	ImportFilterType::IndexType start;
	start.Fill(0);

	ImportFilterType::RegionType region;
	region.SetIndex(start);
	region.SetSize(size);
	importFilter->SetRegion(region);

	const itk::SpacePrecisionType origin[3] = { 0.0, 0.0, 0.0 };
	importFilter->SetOrigin(origin);
	const itk::SpacePrecisionType spacing[3] = { 1.0, 1.0, 1.0 };
	importFilter->SetSpacing(spacing);

	const unsigned int numberOfPixels = size[0] * size[1] * size[2];
	const bool importImageFilterWillOwnTheBuffer = false;
	importFilter->SetImportPointer(buffer, numberOfPixels, importImageFilterWillOwnTheBuffer);

	try {
		importFilter->Update();
	}
	catch (itk::ExceptionObject& error)
	{
		std::cerr << "Error: " << error << std::endl;
	}

	OutputImageType::Pointer outputImg = importFilter->GetOutput();

	return outputImg;

}