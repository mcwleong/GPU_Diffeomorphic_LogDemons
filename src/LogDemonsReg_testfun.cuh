#include <iostream>
#include "LogDemonsReg.cuh"
#include <vector>

class LogDemonsReg_testfun : public LogDemonsReg
{
public:

	LogDemonsReg_testfun(){};
	~LogDemonsReg_testfun(){};

	// Test functions: These functions allocates memories to proper places and writes the test results back to .bin files for testing
	// Ground truth and raw data for testing are stored in separate folder
	void _findupdate();
	void _imgaussian(float sigma);
	void _iminterpolate();
	void _compose();
	void _expfield();
	void _energy();
};

