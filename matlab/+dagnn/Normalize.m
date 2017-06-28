classdef Normalize < dagnn.ElementWise
  properties
    scaleType = 'constant' 
    scaleValue = 20 
    channelShared = false 
    acrossSpatial = false
  end

  methods
    function outputs = forward(obj, inputs, params)

      normalizedFeats = zeros(size(inputs{1}), 'single') ;

      for i = 1:size(inputs{1}, 4)
        if obj.acrossSpatial
            normalizedFeats(:,:,:,i) = inputs{1}(:,:,:,i) ...
                                           / norm(inputs{1}(:,:,:,i), 2) ;
        else
         % TODO: fix for batch size > 1
         normalizedFeats = bsxfun(@rdivide, ...
                                  inputs{1}, ...
                                  sum(inputs{1}.^2, 3) / size(inputs{1}, 3)) ;
        end
        if obj.channelShared
          normalizedFeats(:,:,:,i) = params{1} * normalizedFeats(:,:,:,i) ;
        else
          sz = size(inputs{1});
          multipliers = repmat(params{1}, sz(1:2)) ;
          normalizedFeats(:,:,:,i) = multipliers .* normalizedFeats;
        end
      end
      outputs{1} = single(normalizedFeats) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = {} ;
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = [0,0,0,0] ;
    end

    function rfs = getReceptiveFields(obj)
    end

    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      load@dagnn.Layer(obj, s) ;
    end

    function obj = Normalize(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
