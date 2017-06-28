// @file spp.hpp
// @brief SPP block implementation
// @author Hakan Bilen 

#ifndef VL_NNSPP_H
#define VL_NNSPP_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {


    template<vl::DeviceType dev, typename type>
    struct spp_max {
      typedef type data_type ;

      static vl::ErrorCode
      forward(type* pooled,
              type const* data,
              size_t height, size_t width, size_t depth, size_t size,
              size_t numTotBins,
              size_t numLevels, type const* levels,
              size_t numROIs, type const* ROIs ) ;



      static vl::ErrorCode
      backward(type* derData,
               type const* data,
               type const* derPooled,
               size_t height, size_t width, size_t depth, size_t size,
               size_t numTotBins,
               size_t numLevels, type const* levels,
               size_t numROIs, type const* ROIs) ;
    };

    template<vl::DeviceType dev, typename type>
    struct spp_average {
      typedef type data_type ;

      static vl::ErrorCode
      forward(type* pooled,
              type const* data,
              size_t height, size_t width, size_t depth, size_t size,
              size_t numTotBins,
              size_t numLevels, type const* levels,
              size_t numROIs, type const* ROIs ) ;

      static vl::ErrorCode
      backward(type* derData,
               type const* data,
               type const* derPooled,
               size_t height, size_t width, size_t depth, size_t size,
               size_t numTotBins,
               size_t numLevels, type const* levels,
               size_t numROIs, type const* ROIs) ;
    };
} }
#endif /* defined(VL_NNSPP_H) */
