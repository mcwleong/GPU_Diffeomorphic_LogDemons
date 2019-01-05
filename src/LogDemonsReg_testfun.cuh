#include <iostream>
#include "LogDemonsReg.cuh"
#include <vector>

class LogDemonsReg_testfun : public LogDemonsReg
{
public:

	LogDemonsReg_testfun(){};
	~LogDemonsReg_testfun(){};

	// Test functions: These functions allocates memories to proper places and writes the test results back to .bin files for testing
	void _findupdate();
	void _gradient();
	void _imgaussian(float sigma);
	void _iminterpolate();
	void _compose();
	void _expfield();
	void _jacobian();
};

