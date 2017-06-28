classdef PoolTime < dagnn.Filter
  properties
    method = 'max'
    poolSize = [1 1]
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(obj, inputs, params)
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      nFrames = sz(4) / obj.net.meta.curBatchSize ;
      inputs{1} = reshape(inputs{1}, sz(1)* sz(2), sz(3),  nFrames, sz(4) / nFrames ) ;
      inputs{1} = permute(inputs{1}, [1 3 2 4]);
      if isinf(obj.poolSize(2)), poolT = nFrames; 
      else, poolT= obj.poolSize(2); end % pool all frames
      
      outputs{1} = vl_nnpool(inputs{1}, [obj.poolSize(1) poolT], ...
                             'pad', obj.pad, ...
                             'stride', obj.stride, ...
                             'method', obj.method, ...
                             obj.opts{:}) ;

      outputs{1} = permute(outputs{1}, [1 3 2 4]);
      sz_out = size(outputs{1});
      if numel(sz_out) == 2, sz_out(3) = 1; end % fixes the case of single batch & time pooled
      if numel(sz_out) == 3, sz_out(4) = 1; end % fixes the case of time pooled

      nFrames = sz_out(3);
      obj.net.meta.curNumFrames(obj.layerIndex) = nFrames ;

      sz(1:2) = sz(1:2) / sqrt(obj.stride(1));
      outputs{1} = reshape(outputs{1}, sz(1), sz(2),  sz_out(2),  nFrames*sz_out(4) ) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      sz_in = size(inputs{1}); sz_out = size(derOutputs{1});
      if numel(sz_in) == 3, sz_in(4) = 1; end  
      if numel(sz_out) == 3, sz_out(4) = 1; end
      nFramesIn = sz_in(4) / obj.net.meta.curBatchSize ;
      nFramesOut = sz_out(4) / obj.net.meta.curBatchSize ;

      inputs{1} = reshape(inputs{1}, sz_in(1)* sz_in(2), sz_in(3),  nFramesIn, sz_in(4) / nFramesIn ) ;
      inputs{1} = permute(inputs{1}, [1 3 2 4]);
      derOutputs{1} = reshape(derOutputs{1}, sz_out(1)* sz_out(2), sz_out(3),  nFramesOut, sz_out(4) / nFramesOut ) ;
      derOutputs{1} = permute(derOutputs{1}, [1 3 2 4]);
      if isinf(obj.poolSize(2)), poolT = nFramesIn; 
      else, poolT= obj.poolSize(2); end % pool all frames
      
      derInputs{1} = vl_nnpool(inputs{1}, [obj.poolSize(1) poolT], derOutputs{1}, ...
                               'pad', obj.pad, ...
                               'stride', obj.stride, ...
                               'method', obj.method, ...
                               obj.opts{:}) ;

      derInputs{1} = permute(derInputs{1}, [1 3 2 4]);
      sz_out = size(derInputs{1});
      if numel(sz_out) == 2, sz_out(3) = 1; end % fixes the case of single batch 

      if numel(sz_out) == 3, sz_out(4) = 1; end % fixes the case of single batch 
      nFrames = sz_out(3); % reset nframes
      obj.net.meta.curNumFrames(obj.layerIndex) = nFrames ;

      derInputs{1} = reshape(derInputs{1}, sz_in(1), sz_in(2), sz_out(2), nFrames*sz_out(4) ) ;
      derParams = {} ;

    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = Pooling(varargin)
      obj.load(varargin) ;
    end
  end
end
