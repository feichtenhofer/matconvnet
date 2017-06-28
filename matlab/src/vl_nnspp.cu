// @file spp_cpu.cpp
// @brief SPP block implementation (GPU)
// @author Hakan Bilen 

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnspp.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_numbins=0,
  opt_method,
  opt_verbose,
} ;

/* options */
VLMXOption  options [] = {
  {"NumBins",          1,   opt_numbins },
  {"Method",           1,   opt_method },
  {"Verbose",          0,   opt_verbose },
  {0,                  0,   0           }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_LEVELS, IN_ROIS, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  size_t numLevels  = 0;
  size_t numROIs    = 0;
  size_t numTotBins = 0;
  vl::SPPMethod method = vl::vlSPPMax ;

  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 3) {
    mexErrMsgTxt("The arguments are less than three.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;
      case opt_numbins :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("NUMBINS is not a plain matrix.") ;
        }
        numTotBins = (size_t)mxGetPr(optarg)[0] ;
        break ;
      case opt_method :
        if (!vlmxIsString(optarg,-1)) {
          vlmxError(VLMXE_IllegalArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          method = vl::vlSPPMax ;
        } else if (vlmxIsEqualToStringI(optarg, "avg")) {
          method = vl::vlSPPAverage ;
        } else {
          vlmxError(VLMXE_IllegalArgument, "METHOD is not a supported method.") ;
        }
      default:
        break ;
    }
  }


  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;

  vl::MexTensor rois(context) ;
  vl::MexTensor levels(context) ;

  // load pyramid levels and rois
  levels.init(in[IN_LEVELS]);
  rois.init(in[IN_ROIS]);


  data.init(in[IN_DATA]) ;
  if (backMode) { derOutput.init(in[IN_DEROUTPUT]) ; }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
  }

  numLevels = levels.getNumElements();
  if (numLevels<=0) {
    mexErrMsgTxt("LEVELS has zero elements.") ;
  }

  numROIs = rois.getWidth();

  if ((rois.getHeight() % 5 != 0) || (rois.getNumElements()<=0)) {
    mexErrMsgTxt("ROIs must be a 5xK dimensional array!") ;
  }

  if(numTotBins<=0) {
    mexPrintf("numTotBins %d\n",numTotBins);
    mexErrMsgTxt("numTotBins is wrong.") ;
  }

  if (verbosity > 0) {
    mexPrintf("numTotBins %d depth %d numROIs %d\n",numTotBins,data.getDepth(),numROIs);
  }
  /* Get the output geometry */
  vl::TensorShape outputShape(1, numTotBins,
                              data.getDepth(),
                              numROIs) ;

  vl::TensorShape dataShape = data.getShape();
  dataShape.reshape(4);


  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;


  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, dataShape) ;
  }


  if (verbosity > 0) {
    vl::print("vl_nnspp: data: ", data) ;
    if (backMode) {
      vl::print("vl_nnspp: derOutput: ", derOutput) ;
      vl::print("vl_nnspp: derData: ", derData) ;
    } else {
      vl::print("vl_nnspp: output: ", output) ;
      vl::print("vl_nnspp: rois: ", rois) ;
      vl::print("vl_nnspp: levels: ", levels) ;
    }
  }




  if (verbosity > 0) {
    mexPrintf("vl_spp: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "GPU" : "CPU") ;
    mexPrintf("\nvl_spp: method %d numLevels %d; numROIs %d numTotBins %d\n", method, numLevels, numROIs, numTotBins);
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {
    error = vl::nnspp_forward(context,
                              output, data,
                              method,
                              numTotBins,
                              levels,
                              rois) ;

  } else {
    error = vl::nnspp_backward(context,
                               derData, data, derOutput,
                               method,
                               numTotBins,
                               levels,
                               rois) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
