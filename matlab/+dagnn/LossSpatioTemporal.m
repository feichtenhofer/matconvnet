classdef LossSpatioTemporal < dagnn.GenericLoss
  properties
    loss = 'softmaxlog'
    opts = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      if isfield(obj.net.meta ,'curBatchSize')
        nFrames = sz(4) / obj.net.meta.curBatchSize ;
      else
        nFrames = 1;
      end
      
      inputs{1} = vl_nnsoftmax(inputs{1}) ;
      inputs{2} = vl_nnsoftmax(inputs{2}) ;

      
      outputs{1} = vl_nnpdist(inputs{1}, inputs{2}, 1, 'NoRoot', true, 'aggregate', true) ;
      obj.account(inputs, outputs);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      if isfield(obj.net.meta ,'curBatchSize')
        nFrames = sz(4) / obj.net.meta.curBatchSize ;
      else
        nFrames = 1;
      end
 
            
      [derInputs{1}, derInputs{2}] = vl_nnpdist(inputs{1}, inputs{2}, 1, derOutputs{1}, 'NoRoot', true) ;

      derInputs{1} = vl_nnsoftmax(inputs{1}, derInputs{1}) ;
      derInputs{2} = vl_nnsoftmax(inputs{2}, derInputs{2}) ;

      
      derParams = {} ;
      
    end

    function obj = LossSpatioTemporal(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
