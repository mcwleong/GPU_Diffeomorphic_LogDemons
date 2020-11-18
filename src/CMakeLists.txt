cmake_minimum_required(VERSION 3.19)

  project(CUDA_diffeomorphic_logdemons_3D_ITK LANGUAGES CXX CUDA)
  SET( EX_PLATFORM 64 )
  SET( EX_PLATFORM_NAME "x64" )  
 

 find_package(CUDAToolkit)
  find_package(ITK REQUIRED)
  include(${ITK_USE_FILE})


file(GLOB sources "*.cpp" "*.h" "*.cu" "*.cuh")
add_executable(CUDA_diffeomorphic_logdemons_3D_ITK ${sources})

target_link_libraries(CUDA_diffeomorphic_logdemons_3D_ITK ${ITK_LIBRARIES})  