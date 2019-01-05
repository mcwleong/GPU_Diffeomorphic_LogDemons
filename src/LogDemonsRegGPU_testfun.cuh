#include <iostream>
#include "LogDemonsRegGPU.cuh"
#include <vector>

class LogDemonsRegGPU_testfun : public LogDemonsRegGPU
{
public:

	LogDemonsRegGPU_testfun(){};
	~LogDemonsRegGPU_testfun(){};

	// Test functions: These functions allocates memories to proper places and writes the test results back to .bin files for testing
	void _gradient();
	void _findupdate();
	void _imgaussian(float sigma);
	void _iminterpolate();
	void _compose();
	void _expfield();
	void _jacobian();
	void _energy();
};

