% Concatenated Rectified Linear Units makes an identical copy of the linear
% responses after convolution, negate them, concatenate both
% parts of activation, and then apply Relu altogether.
classdef CReLU < dagnn.ElementWise
  properties
    useShortCircuit = 0
    leak = 0
    opts = {}
  end
  properties (Transient)
    inputSizes = {}
  end


  methods
    function outputs = forward(obj, inputs, params)

      y1 = vl_nnrelu(inputs{1}, [], ...
                             'leak', obj.leak, obj.opts{:}) ;
      y2 = vl_nnrelu(-inputs{1}, [], ...
                             'leak', obj.leak, obj.opts{:}) ;
      outputs{1} = vl_nnconcat({y1,y2}, 3) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      
      obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
      obj.inputSizes = [obj.inputSizes obj.inputSizes];
      % splits derOutputs
      derOutputs = vl_nnconcat(inputs, 3, derOutputs{1}, 'inputSizes', obj.inputSizes) ;

      derInputs{1} = vl_nnrelu(inputs{1}, derOutputs{1}, ...
                               'leak', obj.leak, ...
                               obj.opts{:}) ...
                     - vl_nnrelu(-inputs{1}, derOutputs{2}, ...
                               'leak', obj.leak, ...
                               obj.opts{:}) ;
                             
      derParams = {} ;
    end

    function forwardAdvanced(obj, layer)
      if ~obj.useShortCircuit || ~obj.net.conserveMemory
        forwardAdvanced@dagnn.Layer(obj, layer) ;
        return ;
      end
      net = obj.net ;
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;
      net.vars(out).value = vl_nnrelu(net.vars(in).value, [], ...
                                      'leak', obj.leak, ...
                                      obj.opts{:}) ;
      net.numPendingVarRefs(in) = net.numPendingVarRefs(in) - 1;
      if ~net.vars(in).precious & net.numPendingVarRefs(in) == 0
        net.vars(in).value = [] ;
      end
    end

    function backwardAdvanced(obj, layer)
      if ~obj.useShortCircuit || ~obj.net.conserveMemory
        backwardAdvanced@dagnn.Layer(obj, layer) ;
        return ;
      end
      net = obj.net ;
      in = layer.inputIndexes ;
      out = layer.outputIndexes ;

      if isempty(net.vars(out).der), return ; end

      derInput = vl_nnrelu(net.vars(out).value, net.vars(out).der, ...
                           'leak', obj.leak, obj.opts{:}) ;

      if ~net.vars(out).precious
        net.vars(out).der = [] ;
        net.vars(out).value = [] ;
      end

      if net.numPendingVarRefs(in) == 0
          net.vars(in).der = derInput ;
      else
          net.vars(in).der = net.vars(in).der + derInput ;
      end
      net.numPendingVarRefs(in) = net.numPendingVarRefs(in) + 1 ;
    end

    function obj = CReLU(varargin)
      obj.load(varargin) ;
    end
  end
end
