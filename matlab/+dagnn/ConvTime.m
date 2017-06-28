classdef ConvTime < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end

      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      nFrames = sz(4) / obj.net.meta.curBatchSize ;     
      inputs{1} = reshape(inputs{1}, sz(1)* sz(2), sz(3),  nFrames, sz(4) / nFrames ) ;
      inputs{1} = permute(inputs{1}, [1 3 2 4]);

      outputs{1} = vl_nnconv(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;

      outputs{1} = permute(outputs{1}, [1 3 2 4]);
      sz_out = size(outputs{1});
      if numel(sz_out) == 2, sz_out(3) = 1; end % fixes the case of single batch & time pooled
      if numel(sz_out) == 3, sz_out(4) = 1; end % fixes the case of time pooled

      nFrames = sz_out(3);
      obj.net.meta.curNumFrames(obj.layerIndex) = nFrames ;
      sz(1:2) = floor((sz(1:2) + obj.pad(1:2) - [obj.size(1) obj.size(1)]) / sqrt(obj.stride(1))) + 1 ;
      outputs{1} = reshape(outputs{1}, sz(1), sz(2),  sz_out(2),  nFrames*sz_out(4) ) ;
    end
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      sz_in = size(inputs{1}); sz_out = size(derOutputs{1});
      if numel(sz_in) == 3, sz_in(4) = 1; end  
      if numel(sz_out) == 3, sz_out(4) = 1; end
      nFramesIn = sz_in(4) / obj.net.meta.curBatchSize ;
      nFramesOut = sz_out(4) / obj.net.meta.curBatchSize ;

      inputs{1} = reshape(inputs{1}, sz_in(1)* sz_in(2), sz_in(3),  nFramesIn, sz_in(4) / nFramesIn ) ;
      inputs{1} = permute(inputs{1}, [1 3 2 4]);
      derOutputs{1} = reshape(derOutputs{1}, sz_out(1)* sz_out(2), sz_out(3),  nFramesOut, sz_out(4) / nFramesOut ) ;
      derOutputs{1} = permute(derOutputs{1}, [1 3 2 4]);
      
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
      
      derInputs{1} = permute(derInputs{1}, [1 3 2 4]);
      sz_out = size(derInputs{1});
      if numel(sz_out) == 2, sz_out(3) = 1; end % fixes the case of single frame 
      if numel(sz_out) == 3, 
          sz_out(4) = 1; 
      end % fixes the case of single batch 
      nFrames = sz_out(3); % reset nframes
      obj.net.meta.curNumFrames(obj.layerIndex) = nFrames ;
      derInputs{1} = reshape(derInputs{1}, sz_in(1), sz_in(2), sz_out(2), nFrames*sz_out(4) ) ;
     
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj, sc)
      if nargin < 2
        sc = sqrt(2 / prod(obj.size(1:3))) ;
      end
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') * sc ;
      end
    end

    function params = initParamsDiffTime(obj, sc)
      params{1} = eye(obj.size(3:4),'single')  ;
      if nargin > 1
        params{1} = randn(obj.size(3:4),'single') * sc  ;
      end
      params{1} = permute(params{1}, [4 3 1 2]);
      params{1} = bsxfun(@times, params{1}, [-1 3 -1]);

      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single')  ;
      end
    end
    
    function params = initParamsAvgTime(obj, sc)
      params{1} = eye(obj.size(3:4),'single')  ;
      if nargin > 1
        params{1} = randn(obj.size(3:4),'single') * sc  ;
      end
      params{1} = permute(params{1}, [4 3 1 2]);
      params{1} = bsxfun(@times, params{1}, [1/3 1/3 1/3]);
       
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single')  ;
      end
    end
    function params = initParamsCtrTime(obj)
      params{1} = eye(obj.size(3:4),'single')  ;
      params{1} = permute(params{1}, [4 3 1 2]);
      params{1} = bsxfun(@times, params{1}, [0 1 0]);
       
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single')  ;
      end
    end

    
    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = ConvTime(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
