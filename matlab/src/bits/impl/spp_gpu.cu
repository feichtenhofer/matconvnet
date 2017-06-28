// @file spp_gpu.cu
// @brief  SPP block implementation (GPU)
// @author Hakan Bilen

#include "spp.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>
/* ---------------------------------------------------------------- */
/*                                              spp_average_forward */
/* ---------------------------------------------------------------- */
template<typename T> __global__ void
spp_average_kernel
(T* pooled,
 const T* data,
 const int height,
 const int width,
 const int depth,
 const int size,
 const int numTotBins,
 const int numLevels,
 const T* levels,
 const int numROIs,
 const T* ROIs)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  int pooledVolume = numTotBins * depth * numROIs;

  if (pooledIndex < pooledVolume) {

    int pl = pooledIndex % numTotBins;
    int pc = (pooledIndex / numTotBins) % depth;
    int pr = (pooledIndex / numTotBins / depth); // roi no

    int roi_image   = ROIs[5 * pr + 0];

    int roi_start_h = ROIs[5 * pr + 1];
    int roi_start_w = ROIs[5 * pr + 2];
    int roi_end_h   = ROIs[5 * pr + 3];
    int roi_end_w   = ROIs[5 * pr + 4];

    if(roi_start_w==roi_end_w) {
      if(roi_start_w>0)
        roi_start_w--;
      else
        roi_end_w++;
    }
    if(roi_start_h==roi_end_h) {
      if(roi_start_h>0)
        roi_start_h--;
      else
        roi_end_h++;
    }

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Find pyramid level and bin
    int pb = -1;
    int pLevel = -1;
    int numBins = 0;
    for(int l=0;l<numLevels;l++) {
      if(pl-numBins>=0 && pl-numBins<static_cast<int>(levels[l] * levels[l])) {
        pb = pl - numBins;
        pLevel = l;
      }
      numBins += static_cast<int>(levels[l] * levels[l]);
    }

    int pooledWidth  = levels[pLevel];
    int pooledHeight = levels[pLevel];
    int pw = pb / pooledHeight;
    int ph = pb % pooledHeight;


    const T bin_size_h = static_cast<T>(roi_height)
        / static_cast<T>(pooledHeight);
    const T bin_size_w = static_cast<T>(roi_width)
        / static_cast<T>(pooledWidth);


    int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);

    int offset_data = (roi_image * depth + pc) * (width*height);

    data += offset_data;
    T bestValue = 0;
    const T coef = 1.f / (T)((wend-wstart) * (hend-hstart));
    for (int w = wstart; w < wend; ++w) {
      for (int h = hstart; h < hend; ++h) {
        int index = w * height + h ;
        bestValue += data[index] * coef;
      }
    }
    pooled[pooledIndex] = bestValue ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                  spp_max_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
spp_max_kernel(T* pooled,
 const T* data,
 const int height,
 const int width,
 const int depth,
 const int size,
 const int numTotBins,
 const int numLevels,
 const T* levels,
 const int numROIs,
 const T* ROIs)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;


  int pooledVolume = numTotBins * depth * numROIs;

  if (pooledIndex < pooledVolume) {

    int pl = pooledIndex % numTotBins;
    int pc = (pooledIndex / numTotBins) % depth;
    int pr = (pooledIndex / numTotBins / depth); // roi no
    int roi_image   = ROIs[5 * pr + 0];


    int roi_start_h = ROIs[5 * pr + 1];
    int roi_start_w = ROIs[5 * pr + 2];
    int roi_end_h   = ROIs[5 * pr + 3];
    int roi_end_w   = ROIs[5 * pr + 4];

    if(roi_start_w==roi_end_w) {
      if(roi_start_w>0)
        roi_start_w--;
      else
        roi_end_w++;
    }
    if(roi_start_h==roi_end_h) {
      if(roi_start_h>0)
        roi_start_h--;
      else
        roi_end_h++;
    }


    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Find pyramid level and bin
    int pb = -1;
    int pLevel = -1;
    int numBins = 0;
    for(int l=0;l<numLevels;l++) {
      if(pl-numBins>=0 && pl-numBins<static_cast<int>(levels[l] * levels[l])) {
        pb = pl - numBins;
        pLevel = l;
      }
      numBins += static_cast<int>(levels[l] * levels[l]);
    }

    int pooledWidth  = levels[pLevel];
    int pooledHeight = levels[pLevel];
    int pw = pb / pooledHeight;
    int ph = pb % pooledHeight;


    const T bin_size_h = static_cast<T>(roi_height)
        / static_cast<T>(pooledHeight);
    const T bin_size_w = static_cast<T>(roi_width)
        / static_cast<T>(pooledWidth);


    int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int offset_data = (roi_image * depth + pc) * (width*height);

    data += offset_data;
    T bestValue = is_empty ? 0 : data[wstart * height + hstart];
    for (int w = wstart; w < wend; ++w) {
      for (int h = hstart; h < hend; ++h) {
        int index = w * height + h ;
        bestValue = max(bestValue, data[index]) ;
      }
    }
    pooled[pooledIndex] = bestValue ;

  }
}


/* ---------------------------------------------------------------- */
/*                                                 spp_max_backward */
/* ---------------------------------------------------------------- */

// an implementation of atomicAdd() for double (really slow)
static __device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

template<typename T> __global__ void
spp_max_backward_kernel(T* derData,
                        const T* data,
                        const T* derPooled,
                        const int height,
                        const int width,
                        const int depth,
                        const int size,
                        const int numTotBins,
                        const int numLevels,
                        const T* levels,
                        const int numROIs,
                        const T* ROIs)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;

  int pooledVolume = numTotBins * depth * numROIs;

  if (pooledIndex < pooledVolume) {


    int pl = pooledIndex % numTotBins;
    int pc = (pooledIndex / numTotBins) % depth;
    int pr = (pooledIndex / numTotBins / depth); // roi no

    int roi_image   = ROIs[5 * pr + 0];

    int roi_start_h = ROIs[5 * pr + 1];
    int roi_start_w = ROIs[5 * pr + 2];
    int roi_end_h   = ROIs[5 * pr + 3];
    int roi_end_w   = ROIs[5 * pr + 4];

    if(roi_start_w==roi_end_w) {
      if(roi_start_w>0)
        roi_start_w--;
      else
        roi_end_w++;
    }
    if(roi_start_h==roi_end_h) {
      if(roi_start_h>0)
        roi_start_h--;
      else
        roi_end_h++;
    }

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Find pyramid level and bin
    int pb = -1;
    int pLevel = -1;
    int numBins = 0;
    for(int l=0;l<numLevels;l++) {
      if(pl-numBins>=0 && pl-numBins<static_cast<int>(levels[l] * levels[l])) {
        pb = pl - numBins;
        pLevel = l;
      }
      numBins += static_cast<int>(levels[l] * levels[l]);
    }

    int pooledWidth  = levels[pLevel];
    int pooledHeight = levels[pLevel];
    int pw = pb / pooledHeight;
    int ph = pb % pooledHeight;


    const T bin_size_h = static_cast<T>(roi_height)
        / static_cast<T>(pooledHeight);
    const T bin_size_w = static_cast<T>(roi_width)
        / static_cast<T>(pooledWidth);



    int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    data += (roi_image * depth + pc) * (width*height);
    derData += (roi_image * depth + pc) * (width*height);

    int bestIndex = wstart * height + hstart;
    T bestValue = is_empty ? 0 : data[bestIndex];
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int index = w * height + h ;
        T value = data[index] ;
        if (value > bestValue) {
          bestValue = value ;
          bestIndex = index ;
        }
      }
    }

    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    atomicAdd(derData + bestIndex, derPooled[pooledIndex]) ;
  }
}


/* ---------------------------------------------------------------- */
/*                                             spp_average_backward */
/* ---------------------------------------------------------------- */
template<typename T> __global__ void
spp_average_backward_kernel
(T* derData,
 const T* data,
 const T* derPooled,
 const int height,
 const int width,
 const int depth,
 const int size,
 const int numTotBins,
 const int numLevels,
 const T* levels,
 const int numROIs,
 const T* ROIs)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;


  int pooledVolume = numTotBins * depth * numROIs;

  if (pooledIndex < pooledVolume) {

    int pl = pooledIndex % numTotBins;
    int pc = (pooledIndex / numTotBins) % depth;
    int pr = (pooledIndex / numTotBins / depth); // roi no

    int roi_image   = ROIs[5 * pr + 0];

    int roi_start_h = ROIs[5 * pr + 1];
    int roi_start_w = ROIs[5 * pr + 2];
    int roi_end_h   = ROIs[5 * pr + 3];
    int roi_end_w   = ROIs[5 * pr + 4];

    if(roi_start_w==roi_end_w) {
      if(roi_start_w>0)
        roi_start_w--;
      else
        roi_end_w++;
    }
    if(roi_start_h==roi_end_h) {
      if(roi_start_h>0)
        roi_start_h--;
      else
        roi_end_h++;
    }

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);

    // Find pyramid level and bin
    int pb = -1;
    int pLevel = -1;
    int numBins = 0;
    for(int l=0;l<numLevels;l++) {
      if(pl-numBins>=0 && pl-numBins<static_cast<int>(levels[l] * levels[l])) {
        pb = pl - numBins;
        pLevel = l;
      }
      numBins += static_cast<int>(levels[l] * levels[l]);
    }
    int pooledWidth  = levels[pLevel];
    int pooledHeight = levels[pLevel];
    int pw = pb / pooledHeight;
    int ph = pb % pooledHeight;


    const T bin_size_h = static_cast<T>(roi_height)
        / static_cast<T>(pooledHeight);
    const T bin_size_w = static_cast<T>(roi_width)
        / static_cast<T>(pooledWidth);


    int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);

    data += (roi_image * depth + pc) * (width*height);
    derData += (roi_image * depth + pc) * (width*height);

    const T coef = 1.f / (T)((wend-wstart)*(hend-hstart));
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int index = w * height + h ;
      /*
       This is bad, but required to eliminate a race condition when writing
       to bottom_diff.
       Caffe goes the other way around, but requires remembering the layer
       output, or the maximal indexes.
       atomicAdd(add, val)
       */
        atomicAdd(derData + index, derPooled[pooledIndex] * coef) ;
      }
    }


  }
}
/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */
namespace vl { namespace impl {

  template <typename type>
  struct spp_max<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t depth, size_t size,
            size_t numTotBins,
            size_t numLevels, type const* levels,
            size_t numROIs, type const* ROIs)
  {
    int pooledVolume = numTotBins * depth * numROIs;

    spp_max_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      (pooled, data,
       height, width, depth, size,
       numTotBins,
       numLevels, levels,
       numROIs, ROIs);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* derPooled,
             size_t height, size_t width, size_t depth, size_t size,
             size_t numTotBins,
             size_t numLevels, type const* levels,
             size_t numROIs, type const* ROIs)
    {
      int pooledVolume = numTotBins * depth * numROIs;

      spp_max_backward_kernel<type>
          <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
          (derData, data, derPooled,
           height, width, depth, size,
           numTotBins,
           numLevels, levels,
           numROIs, ROIs);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // spp_max

  template <typename type>
  struct spp_average<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t depth, size_t size,
            size_t numTotBins,
            size_t numLevels, type const* levels,
            size_t numROIs, type const* ROIs)
  {
    int pooledVolume = numTotBins * depth * numROIs;

    spp_average_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS),VL_CUDA_NUM_THREADS >>>
      (pooled, data,
       height, width, depth, size,
       numTotBins,
       numLevels, levels,
       numROIs, ROIs);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* derPooled,
             size_t height, size_t width, size_t depth, size_t size,
             size_t numTotBins,
             size_t numLevels, type const * levels,
             size_t numROIs, type const * ROIs)
    {
      int pooledVolume = numTotBins * depth * numROIs;

      spp_average_backward_kernel<type>
          <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
          (derData, data, derPooled,
           height, width, depth, size,
           numTotBins,
           numLevels, levels,
           numROIs, ROIs);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // spp_average
} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::spp_max<vl::VLDT_GPU, float> ;
template struct vl::impl::spp_average<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::spp_max<vl::VLDT_GPU, double> ;
template struct vl::impl::spp_average<vl::VLDT_GPU, double> ;
#endif
