classdef LossSmoothL1 < dagnn.Loss
  properties
    sigma2 = 1;
  end
%   properties (Transient)
%     average = 0
%     numAveraged = 0
%   end
  methods
%   f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
%   |x| - 0.5 / sigma / sigma    otherwise
    function outputs = forward(obj, inputs, params)
      
      dif = inputs{1} - inputs{2};
      abs_input = abs(dif);
            
      idx1 = (abs_input > 1 / obj.sigma2);
      
      t = zeros(size(abs_input),'like',abs_input);
      t(idx1) = abs_input(idx1) - 0.5 / obj.sigma2;
      t(~idx1) = 0.5 * obj.sigma2 * dif(~idx1) .^2 ;
      
      if numel(inputs) < 3, instance_weights = ones(size(t),'like',t);
      else
        instance_weights = inputs{3};
      end
      outputs{1} = sum(instance_weights(:) .* t(:));
      n = obj.numAveraged ;
      
      m = n + gather(sum(instance_weights(:))) + 1e-9;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end
%     f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
%            = sign(x)                   otherwise
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,numel(inputs));

      dif = inputs{1} - inputs{2};
      abs_input = abs(dif);
      
      t = obj.sigma2 * dif;
      idx1 = (abs_input > 1 / obj.sigma2);
      
      t(idx1) = sign(dif(idx1));

      if numel(inputs) < 3, instance_weights = ones(size(t),'like',t);
      else
        instance_weights = inputs{3};
      end
      
      derInputs{1} = instance_weights .* t .* derOutputs{1};
      if numel(inputs) == 2
        derInputs{2} = -instance_weights .* t .* derOutputs{1};
      end
      derParams = {} ;
    end
    
    function obj = LossSmoothL1(varargin)
      obj.load(varargin) ;
      obj.loss = 'smoothl1';
    end
    
    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end
  end
end
