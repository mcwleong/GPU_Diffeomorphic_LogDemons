#include <fstream>
#include <string>
#include <bitset>
#include <iostream>

using namespace std;



template <typename T>
void saveSlice(T* const data,  unsigned int dim[3], string filename) {
	int	len = dim[0] * dim[1] * dim[2];
	short dims[3] = { (short)dim[0], (short)dim[1], (short)dim[2] };
	T* ptr = data;

	FILE *file = fopen(filename.c_str(), "wb");
	cout << "outputfile: " << filename << endl;
	//cout << "size of output data bit" << sizeof(*ptr) << endl;
	//cout << "Output Length: " << len << endl;


	fwrite((void*)&dims, sizeof(short), 3, file);
	fwrite(ptr, sizeof(T), len, file);

	fclose(file);
};

template <typename T>
void loadBin(T* &dataptr, string filename )
{
	FILE *file = fopen(filename.c_str(), "rb");
	short dim[3];
	fread(&dim, sizeof(short), 3, file);
	int len = dim[0] * dim[1] * dim[2];
	cout << "Reading file with length of: " << len << endl;

	//dataptr = new T[len];
	fread(dataptr, sizeof(T), len, file);
	fclose(file);
}


