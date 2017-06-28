classdef SpatialL2Norm < dagnn.ElementWise
  properties
    param = [2 2 10 2]
  end

  methods
    function outputs = forward(obj, inputs, params)
      inputs{1} = vl_nnsqrt(inputs{1});
      outputs{1} = vl_nnl2norm( inputs{1});
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      sz = size(derOutputs{1});
      derInputs{1}  = vl_nnl2norm( inputs{1}, derOutputs{1});
      derInputs{1} = vl_nnsqrt(inputs{1}, derInputs{1});
      derParams = {} ;
    end

    function obj = SpatialNorm(varargin)
      obj.load(varargin) ;
    end
  end
end
