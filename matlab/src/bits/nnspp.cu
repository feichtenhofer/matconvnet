// @file nnspp.cu
// @brief SPP block
// @author Hakan Bilen 

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/


#include "nnspp.hpp"
#include "impl/spp.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                nnspp_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, op, type) \
  status = vl::impl::op<deviceType, type>::forward \
  ((type*)output.getMemory(), (type const*)data.getMemory(), \
  data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
  numTotBins,\
  levels.getNumElements(), (type const*)levels.getMemory(),\
  ROIs.getWidth(), (type const *)ROIs.getMemory()) ;


#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, op, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, op, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCH3(deviceType) \
  switch (method) { \
  case vlSPPAverage : DISPATCH2(deviceType, spp_average) ; break ; \
  case vlSPPMax : DISPATCH2(deviceType, spp_max) ; break ; \
  default: assert(false) ; return VLE_Unknown ; \
  }

ErrorCode
vl::nnspp_forward(vl::Context& context,
                  vl::Tensor output,
                  vl::Tensor data,
                  size_t method,
                  size_t numTotBins,
                  vl::Tensor levels,
                  vl::Tensor ROIs)
{
  ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = output.getDeviceType();
  vl::DataType dataType = output.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

  case vl::VLDT_CPU:
	  DISPATCH3(vl::VLDT_CPU);
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
		DISPATCH3(VLDT_GPU) ;
      if (status == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnspp_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                                   nnspp_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

// backward max and average want slightly differet argument lists

#define DISPATCH_spp_average(deviceType, type) \
  status = vl::impl::spp_average<deviceType, type>::backward \
  ((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)derPooled.getMemory(), \
  derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(), \
  numTotBins, \
  levels.getNumElements(), (const type *)levels.getMemory(), \
  ROIs.getWidth(), (const type *)ROIs.getMemory()) ; \

#define DISPATCH_spp_max(deviceType, type) \
  status = vl::impl::spp_max<deviceType, type>::backward \
  ((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)derPooled.getMemory(), \
  derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(), \
  numTotBins, \
  levels.getNumElements(), (const type *)levels.getMemory(), \
  ROIs.getWidth(), (const type *)ROIs.getMemory());

#define DISPATCH2(deviceType, op) \
  switch (dataType) { \
  case VLDT_Float : DISPATCH_ ## op (deviceType, float) ; break ; \
  IF_DOUBLE(case VLDT_Double : DISPATCH_ ## op (deviceType, double) ; break ;) \
  default: assert(false) ; return VLE_Unknown ; \
  }

ErrorCode
vl::nnspp_backward(Context& context,
                   Tensor derData,
                   Tensor data,
                   Tensor derPooled,
                   size_t method,
                   size_t numTotBins,
                   Tensor levels,
                   Tensor ROIs)
{
  vl::ErrorCode status = VLE_Success;
  vl::DeviceType deviceType = derPooled.getDeviceType() ;
  vl::DataType dataType = derPooled.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
	  return vl::VLE_Unknown;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
	case vl::VLDT_GPU:
		DISPATCH3(vl::VLDT_GPU) ;
      if (status == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("spp_*::backward")) ;
      }
      break ;
#endif
  }

  return context.passError(status, "nnspp_backward: ") ;
}
