classdef Loss < dagnn.ElementWise
  properties
    loss = 'softmaxlog'
    ignoreAverage = false
    opts = {}
    hard_temporal_mining = 0
  end

  properties (Transient)
    average = 0
    numAveraged = 0
    hard_temporal_idx = []
  end

  methods
    function outputs = forward(obj, inputs, params)
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      if isfield(obj.net.meta ,'curBatchSize')
        nFrames = sz(4) / obj.net.meta.curBatchSize ;
      else
        nFrames = 1;
      end
      if numel(inputs) == 1, inputs{2} = zeros(size(sz(4),1 )); end

      if any(sz(1:2) > 1) || nFrames > 1
        if strfind(obj.loss, 'error') % average predictions over frames and spatial locations
          f_idx = 1:nFrames:size(inputs{1},4);
          pred_avg = {}; pred_max = {};
          for i = 1:size(f_idx,2)
            pred_avg{i} = mean(mean(mean(inputs{1}(:,:,:,f_idx(i):f_idx(i)+nFrames-1),1),2),4);
          end
          inputs{1} = cat(4,pred_avg{:});

        else % replicate labels for all frames
          inputs{2} = repmat(inputs{2}, nFrames, 1);
          inputs{2} = inputs{2}(:);
        end
      end

      layername = obj.net.layers(obj.layerIndex).name ;
      if obj.hard_temporal_mining && strcmp(obj.net.mode, 'normal')           
        outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}, ...
          'aggregate',false) ;

        frames = 1:nFrames;
        frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
        batches = [0 cumsum(repmat(nFrames,1,obj.net.meta.curBatchSize-1))] + 1;
        offsets = repmat(batches,nFrames,1) + frames - 1 ;
        for k=1:numel(batches)
          [maxL, maxIdx] = max(outputs{1}(:,:,:,offsets(:,k)),[], 4);
          obj.hard_temporal_idx = [obj.hard_temporal_idx maxIdx ];
        end
        outputs{1} = sum(outputs{1}(:));
      else
        outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, obj.opts{:}) ;
      end

      obj.accumulateAverage(inputs, outputs);
    end

    function accumulateAverage(obj, inputs, outputs)
      if obj.ignoreAverage, return; end;
      n = obj.numAveraged ;
      m = n + size(inputs{1}, 1) *  size(inputs{1}, 2) * size(inputs{1}, 4);
      obj.average = bsxfun(@plus, n * obj.average, gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      sz = size(inputs{1}); if numel(sz) < 4, sz(4) = 1; end 
      if isfield(obj.net.meta ,'curBatchSize')
        nFrames = sz(4) / obj.net.meta.curBatchSize ;
      else
        nFrames = 1;
      end
      
      derParams = {} ;
      if numel(inputs) == 1, inputs{2} = zeros(size(sz(4),1 ));
      else
        derInputs{2} = [] ;
      end

       if nFrames > 1 
        if strfind(obj.loss, 'error') % average predictions over frames
          f_idx = 1:nFrames:size(inputs{1},4);
          pred_avg = {};
          for i = 1:size(f_idx,2)
            pred_avg{i} = mean(mean(mean(inputs{1}(:,:,:,f_idx(i):f_idx(i)+nFrames-1),1),2),4);
          end
          inputs{1} = cat(4,pred_avg{:});
        else % replicate labels for all frames
          inputs{2} = repmat(inputs{2}, nFrames, 1);
          inputs{2} = inputs{2}(:);
        end
       end
       
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, obj.opts{:}) ;

      if obj.hard_temporal_mining && strcmp(obj.net.mode, 'normal')           
        frames = 1:nFrames;
        frames = [repmat(frames',1,obj.net.meta.curBatchSize) ] ;
        batches = [0 cumsum(repmat(nFrames,1,obj.net.meta.curBatchSize-1))] + 1;
        offsets = repmat(batches,nFrames,1) + frames - 1 ;
        offsets(obj.hard_temporal_idx + batches - 1) = [];
        derInputs{1}(:,:,:,offsets) = derInputs{1}(:,:,:,offsets) .* 0;
        obj.hard_temporal_idx = [];
      end
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
      rfs(3,1) = rfs(1,1) ;
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
