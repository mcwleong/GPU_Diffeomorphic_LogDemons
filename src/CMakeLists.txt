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
MESSAGE("HELLO")
add_compile_options("/wd4819")

FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(GDCM REQUIRED)
    IF(GDCM_FOUND)
        INCLUDE(${GDCM_USE_FILE})
        SET(GDCM_LIBRARIES 
            gdcmcharls
            gdcmCommon
            gdcmDICT
            gdcmDSED
            gdcmexpat
            gdcmgetopt
            gdcmIOD
            gdcmjpeg12
            gdcmjpeg16
            gdcmjpeg8
            gdcmMEXD
            gdcmMSFF
            gdcmopenjpeg
            gdcmzlib
            socketxx
            Rpcrt4)
    ELSE(GDCM_FOUND)
        MESSAGE(FATAL_ERROR "Cannot find GDCM, did you set GDCM_DIR?")
    ENDIF(GDCM_FOUND)

SET(GDCM_DIR "H:/diffeomorphic_cuda/Lib/gdcm_build")

SET(CMAKE_EXE_LINKER_FLAGS " ${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

file(GLOB sources "*.cpp" "*.h" "*.cu" "*.cuh")
cuda_add_executable(cudaDiffeomorphic3D  ${sources})
TARGET_LINK_LIBRARIES(cudaDiffeomorphic3D ${GDCM_LIBRARIES})

