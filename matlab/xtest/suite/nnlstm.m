classdef nnlstm < nntest
  
  properties (TestParameter)
    N = {1,2,4};
    Din = {1,2,4};
    Dout = {1,2,4};
    Dlstm = {1,2,4};
  end

  methods (Test)
    % note the parameters are provided by the test class
    % by trying out the Cartesian product of the test-parameters.
    % N is the batch-size:
    % check the forward/backward pass (gradients) of the LSTM:
    function check_Dx(test, Din, Dout, Dlstm, N)
      [inputs,params,grads,outputs,grad_out] = test.getParams(Din, Dout, Dlstm, N);
      [x,mask,hp,cp] = inputs{:};
      [Dx,Dhp,Dcp,DWi,Dbi,DWo,Dbo] = grad_out{:};
      test.deriv(@(inp) vl_nnlstm({inp,mask,hp,cp}, params, {}), x, grads(1:3), Dx, 1e-2);
    end

    function check_Dhp(test, Din, Dout, Dlstm, N)
      [inputs,params,grads,outputs,grad_out] = test.getParams(Din, Dout, Dlstm, N);
      [x,mask,hp,cp] = inputs{:};
      [Dx,Dhp,Dcp,DWi,Dbi,DWo,Dbo] = grad_out{:};
      test.deriv(@(inp) vl_nnlstm({x,mask,inp,cp}, params, {}), hp, grads(1:3), Dhp, 1e-2);
    end

    function check_Dcp(test, Din, Dout, Dlstm, N)
      [inputs,params,grads,outputs,grad_out] = test.getParams(Din, Dout, Dlstm, N);
      [x,mask,hp,cp] = inputs{:};
      [Dx,Dhp,Dcp,DWi,Dbi,DWo,Dbo] = grad_out{:};
      test.deriv(@(inp) vl_nnlstm({x,mask,hp,inp}, params, {}), cp, grads(1:3), Dcp, 1e-2);
    end

    function check_DWi(test, Din, Dout, Dlstm, N)
      [inputs,params,grads,outputs,grad_out] = test.getParams(Din, Dout, Dlstm, N);
      [Dx,Dhp,Dcp,DWi,Dbi,DWo,Dbo] = grad_out{:};
      [Wi,bi,Wo,bo] = params{:};
      test.deriv(@(inp) vl_nnlstm(inputs, {inp,bi,Wo,bo}, {}), Wi, grads(1:3), DWi, 1e-2);
    end

    function check_Dbi(test, Din, Dout, Dlstm, N)
      [inputs,params,grads,outputs,grad_out] = test.getParams(Din, Dout, Dlstm, N);
      [Dx,Dhp,Dcp,DWi,Dbi,DWo,Dbo] = grad_out{:};
      [Wi,bi,Wo,bo] = params{:};
      test.deriv(@(inp) vl_nnlstm(inputs, {Wi,inp,Wo,bo}, {}), bi, grads(1:3), Dbi, 1e-2);
    end

    function check_DWo(test, Din, Dout, Dlstm, N)
      [inputs,params,grads,outputs,grad_out] = test.getParams(Din, Dout, Dlstm, N);
      [Dx,Dhp,Dcp,DWi,Dbi,DWo,Dbo] = grad_out{:};
      [Wi,bi,Wo,bo] = params{:};
      test.deriv(@(inp) vl_nnlstm(inputs, {Wi,bi,inp,bo}, {}), Wo, grads(1:3), DWo, 1e-2);
    end

    function check_Dbo(test, Din, Dout, Dlstm, N)
      [inputs,params,grads,outputs,grad_out] = test.getParams(Din, Dout, Dlstm, N);
      [Dx,Dhp,Dcp,DWi,Dbi,DWo,Dbo] = grad_out{:};
      [Wi,bi,Wo,bo] = params{:};
      test.deriv(@(inp) vl_nnlstm(inputs, {Wi,bi,Wo,inp}, {}), bo, grads(1:3), Dbo, 1e-2);
    end
  end

  methods 
    function deriv(obj, g, x, DZs, Dx_compare, delta, tau)
      if nargin < 7
        tau = [] ;
      end
      Dx_numdiff = obj.toDataType(obj.numder_multiout(g, x, DZs, delta)) ;
      obj.eq(gather(Dx_numdiff), gather(Dx_compare), tau);
    end

    function [inputs,params,grads,outputs,grad_out]=getParams(test, Din, Dout, Dlstm, N)
      test.range = 1;
      % setup inputs:
      x = test.randn(1,1,Din,N);
      mask = test.ones(1,1,1,N);
      hp = test.randn(1,1,Dlstm,N);
      cp = test.randn(1,1,Dlstm,N);
      inputs = {x,mask,hp,cp};

      % setup params:
      Wi = test.randn(4*Dlstm,Din+Dlstm);
      bi = 10*test.randn(4*Dlstm,1); 
      bi(Dlstm+1:2*Dlstm,:) = test.randn(Dlstm,1); 
      
      Wo = test.randn(Dout,Dlstm);
      bo = 10*test.randn(Dout,1);
      params = {Wi,bi,Wo,bo};

      % setup grads:
      DzDy = test.randn(1,1,Dout,N);
      DzDhn = test.randn(1,1,Dlstm,N);
      DzDcn = test.randn(1,1,Dlstm,N);
      grads = {DzDy, DzDhn, DzDcn};

      outputs = vl_nnlstm(inputs, params, {});

      grad_out = vl_nnlstm(inputs, params, grads);
      %[Dx,Dhp,Dcp,DWi,Dbi,DWo,Dbo] = grad_out{:};
    end
  end

  methods (Static)
    function dzdx = numder_multiout(g, x, DZs, delta)
      if nargin < 3
        delta = 1e-3;
      end
      DZs = cellfun(@gather, DZs, 'UniformOutput', false); % multiple-gradients
      y = cellfun(@gather, g(x), 'UniformOutput', false); % multiple-outputs

      dzdx = zeros(size(x),'double') ;
      for i=1:numel(x)
        x_ = x ;
        x_(i) = x_(i) + delta ;
        y_ = cellfun(@gather, g(x_), 'UniformOutput', false);

        x_(i) = x_(i) - 2*delta;
        yp = cellfun(@gather, g(x_), 'UniformOutput', false);

        factors = cellfun(@(up,u,dzdu) sign(dzdu.*(up - u)).*exp(log(abs(dzdu.*(up - u)))-log(2*delta)),  y_, yp, DZs, 'UniformOutput', false);
        sum_factors = 0;
        for j=1:numel(factors)
          fj = factors{j};
          dzdx(i) = dzdx(i) + sum(fj(:));
        end
      end
    end
  end

end


