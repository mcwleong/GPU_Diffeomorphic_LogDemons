Open-source C++/CUDA implementation of the Diffeomorphic Log-Demons method for non rigid image registration.

Installation steps to use the ITK wrapper:
1.	Install ITK: https://itk.org/download/
2.	Amend the dicom directory in the src/main.cpp depends on your OS
3.	Compile the 'src' source code to 'build' directory using CMake
4.	Specify the ITK path in the configuration 
5. 	Run 'CUDA_diffeomorphic_logdemons_3D_ITK' under the 'build' directory


The work is linked to the following publication (to appear): 
Performance-aware Programming for Intraoperative Intensity-based Image Registration on Graphics Processing Units
Martin C.W. Leong, Kit-Hang Lee, Bowen P.Y. Kwan, Yui-Lun Ng, Zhiyu Liu, Nassir Navab, Wayne Luk, Ka-Wai Kwok

If you have any question, please feel free to contact me at mcwleong@connect.hku.hk or mcwleong1993@gmail.com

