CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(CUDA_diffeomorphic_logdemons_3D)

if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
    MESSAGE( "64 bits compiler detected" )
    SET( EX_PLATFORM 64 )
    SET( EX_PLATFORM_NAME "x64" )
else( CMAKE_SIZEOF_VOID_P EQUAL 8 ) 
    MESSAGE( "32 bits compiler detected" )
    SET( EX_PLATFORM 32 )
    SET( EX_PLATFORM_NAME "x86" )
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 ) 

if (EX_PLATFORM=64)
	set( MSVC_FLAGS "${MSVC_FLAGS} -DWIN64" )
endif ()
add_compile_options("/wd4819")

FIND_PACKAGE(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math" )
SET(CMAKE_EXE_LINKER_FLAGS " ${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE") 

file(GLOB sources "*.cpp" "*.h" "*.cu" "*.cuh" )
MESSAGE(${sources})
cuda_add_executable(CUDA_diffeomorphic_logdemons_3D ${sources})

 