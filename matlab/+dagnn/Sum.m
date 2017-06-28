classdef Sum < dagnn.ElementWise
  %SUM DagNN sum layer
  %   The SUM layer takes the sum of all its inputs and store the result
  %   as its only output.

  properties (Transient)
    numInputs
    inputSizes
  end
  properties 
    pickTemporalCentre = true ;
  end
  methods
    function outputs = forward(obj, inputs, params)
      obj.numInputs = numel(inputs) ;
      obj.inputSizes = cellfun(@(x) size(x,4), inputs, 'UniformOutput', false) ;
      if ~isequaln(obj.inputSizes{:}) && obj.pickTemporalCentre
        nFrames = cat(1, obj.inputSizes{:});
        nFrames =  nFrames / obj.net.meta.curBatchSize ;
        [minFrames, idx] = min(nFrames);
        for k = 1:obj.numInputs
          if k == idx, continue; end
            batches = [0 cumsum(repmat(nFrames(k),1,obj.net.meta.curBatchSize-1))] + 1;
            frames = round(linspace(1,nFrames(k), minFrames));
            frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
            offsets = repmat(batches,minFrames,1) + frames - 1 ;
            inputs{k} = inputs{k}(:,:,:,offsets) ;
        end
      end
      outputs{1} = inputs{1} ;
      for k = 2:obj.numInputs
        outputs{1} = outputs{1} + inputs{k} ;
      end

    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      
      if ~isequaln(obj.inputSizes{:}) && obj.pickTemporalCentre
        nFrames = cat(1, obj.inputSizes{:});
        nFrames =  nFrames / obj.net.meta.curBatchSize ;
        [minFrames, idx] = min(nFrames);
        derInputs{idx} = derOutputs{1} ;
        for k = 1:obj.numInputs
          if k == idx, continue; end
            derInputs{k} = zeros(size(inputs{k}),'like',inputs{k}); 
            batches = [0 cumsum(repmat(nFrames(k),1,obj.net.meta.curBatchSize-1))] + 1;
            frames = round(linspace(1,nFrames(k), minFrames));
            frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
            offsets = repmat(batches,minFrames,1) + frames - 1 ;
            derInputs{k}(:,:,:,offsets) = derOutputs{1} ;
        end
      else
        for k = 1:obj.numInputs
          derInputs{k} = derOutputs{1} ;
        end
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

    function obj = Sum(varargin)
      obj.load(varargin) ;
    end
    

  end
end
