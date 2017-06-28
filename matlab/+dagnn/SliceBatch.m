classdef SliceBatch < dagnn.ElementWise
  properties
    numOut 
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs = cell(numel(obj.net.layers(obj.layerIndex).outputIndexes),1);
      if numel(inputs{2})==1, sets = inputs{2}(1); else, sets = unique(inputs{2}); end;
      for o = sets
        k = 2 * o - 1;
        outputs{k} = inputs{1}(:,:,:,inputs{2}==o);
        outputs{k+1} = inputs{3}(inputs{2}==o);
      end
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if numel(inputs{2})==1, sets = inputs{2}(1); else, sets = unique(inputs{2}); end;
      derInputs{1} = [];  
       i_set = ones(gather(max(inputs{2})),1);
      for set = inputs{2}
        derInputs{1} = cat(4,derInputs{1}, derOutputs{2 * set - 1}(:,:,:,i_set(set)) );
        i_set(set) = i_set(set)+1;
      end
      derInputs{2} = [];
      derInputs{3} = [];
      derParams = {} ;
    end

    function reset(obj)
      obj.inputSizes = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      sz = inputSizes{1} ;
      outputSizes{1} = sz ;
    end

    function rfs = getReceptiveFields(obj)
      if obj.dim == 3 || obj.dim == 4
        rfs = getReceptiveFields@dagnn.ElementWise(obj) ;
        rfs = repmat(rfs, obj.numInputs, 1) ;
      else
        for i = 1:obj.numInputs
          rfs(i,1).size = [NaN NaN] ;
          rfs(i,1).stride = [NaN NaN] ;
          rfs(i,1).offset = [NaN NaN] ;
        end
      end
    end

    function obj = Concat(varargin)
      obj.load(varargin) ;
    end
  end
end
