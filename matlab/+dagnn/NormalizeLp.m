classdef NormalizeLp < dagnn.ElementWise
  properties
    epsilon = 0.01
    p = 2
    spatial = false
    standardize = false
    subtractMean = false
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnnormalizelp(inputs{1}, [], ...
                                    'spatial', obj.spatial, ...
                                    'p', obj.p, ...
                                    'epsilon', obj.epsilon, ...
                                    'standardize', obj.standardize, ...
                                    'subtractMean', obj.subtractMean) ;
    end

   function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnnormalizelp(inputs{1}, derOutputs{1}, ...
                                      'spatial', obj.spatial, ...
                                      'p', obj.p, ...
                                      'epsilon', obj.epsilon, ...
                                      'standardize', obj.standardize, ...
                                      'subtractMean', obj.subtractMean) ;
      derParams = {} ;
    end

    function obj = NormOffset(varargin)
      obj.load(varargin) ;
    end
  end
end
