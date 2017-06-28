classdef MulTime < dagnn.ElementWise
  properties
    numInputs
    backPropInput = []
    denseOutput = false;
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
      
      if obj.denseOutput
          outputs{1} = zeros(sz_in,'like',inputs{1}); 

          frames = 2:nFrames-1;
          frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
          offsets = repmat(batches,nFrames-2,1) + frames - 1 ;

          outputs{1}(:,:,:,offsets) = inputs{1}(:,:,:,offsets) .* ... 
                 inputs{1}(:,:,:,offsets+1) +  ... 
                 inputs{1}(:,:,:,offsets) .* ... 
                 inputs{1}(:,:,:,offsets-1) ;

           % border frames     
          outputs{1}(:,:,:,batches) = (inputs{1}(:,:,:,batches) .* ... 
                 inputs{1}(:,:,:,batches+1)) .* 2 ; 
          outputs{1}(:,:,:,batches+nFrames-1) = (inputs{1}(:,:,:,batches+nFrames-1) .* ... 
                 inputs{1}(:,:,:,batches+nFrames-2)) .* 2 ;


      else
          assert(~mod(nFrames,2))

          frames = 1:2:nFrames;
          frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
          offsets = repmat(batches,nFrames/2,1) + frames - 1 ;
          outputs{1} = inputs{1}(:,:,:,offsets) .* ... 
                       inputs{1}(:,:,:,offsets+1);
      end
      % hotfix in case relu follows the input and optimizes out
      obj.net.numPendingVarRefs(obj.net.layers(obj.layerIndex).inputIndexes) = ...
             obj.net.numPendingVarRefs(obj.net.layers(obj.layerIndex).inputIndexes) + 1;
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
      derInputs{1} = zeros(sz_in,'like',inputs{1}); 

      if obj.denseOutput
        frames = 2:nFrames-1;
        frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
        offsets = repmat(batches,nFrames-2,1) + frames - 1 ;
        derInputs{1}(:,:,:,offsets) = inputs{1}(:,:,:,offsets+1) .* derOutputs{1}(:,:,:,offsets) + ...;
                                      inputs{1}(:,:,:,offsets-1) .* derOutputs{1}(:,:,:,offsets) ; 
      
         % border frames     
        derInputs{1}(:,:,:,batches) = (derOutputs{1}(:,:,:,batches) .* ... 
               inputs{1}(:,:,:,batches+1)) .* 2 ; 
        derInputs{1}(:,:,:,batches+nFrames-1) = (derOutputs{1}(:,:,:,batches+nFrames-1) .* ... 
               inputs{1}(:,:,:,batches+nFrames-2)) .* 2 ;
      else
        frames = 1:2:nFrames;
        frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
        offsets = repmat(batches,nFrames/2,1) + frames - 1 ;

        derInputs{1}(:,:,:,offsets) = inputs{1}(:,:,:,offsets+1) .* derOutputs{1};
        derInputs{1}(:,:,:,offsets+1) =  derInputs{1}(:,:,:,offsets+1) + ...
                                      inputs{1}(:,:,:,offsets) .* derOutputs{1};
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

    function obj = MulTime(varargin)
      obj.load(varargin) ;
    end
    
    function y = zerosLike(x)
      if isa(x,'gpuArray')
        y = gpuArray.zeros(size(x),classUnderlying(x)) ;
      else
        y = zeros(size(x),'like',x) ;
      end
    end
  end
end
