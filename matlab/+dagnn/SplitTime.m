classdef SplitTime < dagnn.ElementWise

  properties (Transient)
    numInputs
    inputSizes
    numOutputs
  end

  methods
    function outputs = forward(obj, inputs, params)
      obj.numInputs = numel(inputs) ;
        
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      if isfield(obj.net.meta ,'curBatchSize')
        nFrames = sz(4) / obj.net.meta.curBatchSize ;
      else
        nFrames = 1;
      end
      sz_in = size(inputs{1});
      
      batches = [0 cumsum(repmat(nFrames,1,obj.net.meta.curBatchSize-1))] + 1;

      frames = 1:nFrames;
      frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
      offsets = repmat(batches,nFrames,1) + frames - 1 ;
      
      
      obj.numOutputs = nFrames ;
      outputs = {};
      for k = 1:obj.numOutputs
        outputs{k} = inputs{1}(:,:,:,offsets(k,:));
      end
    
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      if isfield(obj.net.meta ,'curBatchSize')
        nFrames = sz(4) / obj.net.meta.curBatchSize ;
      else
        nFrames = 1;
      end
      sz_in = size(inputs{1});

      batches = [0 cumsum(repmat(nFrames,1,obj.net.meta.curBatchSize-1))] + 1;
      
      if nFrames > 1
        derInputs{1} = cat(5, derOutputs{:});
        derInputs{1} = permute(derInputs{1}, [1 2 3 5 4]);
        derInputs{1} = reshape(derInputs{1}, sz_in);
      else
        derInputs{1} = derOutputs{1} ;
      end
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Sum layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = DiffTime(varargin)
      
      obj.load(varargin) ;
    end
    

  end
end
