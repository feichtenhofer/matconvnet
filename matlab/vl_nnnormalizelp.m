function y = vl_nnnormalizelp(x,dzdy,varargin)
%VL_NNNORMALIZELP  CNN Lp normalization
%   Y = VL_NNNORMALIZELP(X) normalizes each column of the tensor X
%   in L^p norm:
%
%       Y(i,j,k) = X(i,j,k) / sum_q (X(i,j,q).^p + epsilon)^(1/p)
%
%   DZDX = VL_NNNORMALIZELP(X, DZDY) computes the derivative of the
%   function with respect to X projected onto DZDY.
%
%   VL_NNNORMALIZE(___, 'opts', val, ...) takes the following options:
%
%   `P`:: 2
%      The exponent of the Lp norm. Warning: currently only even
%      exponents are supported.
%
%   `Epsilon`:: 0.01
%      The constant added to the sum of p-powers before taking the
%      1/p square root (see the formula above).
%
%   `Spatial`:: `false`
%      If `true`, sum along the two spatial dimensions instead of
%      along the feature channels.
%
%   `SubtractMean`:: `false`
%      If `true`, preprocess the data by subtracting its mean.  Means
%      are computed along columns or across spatial locations (for
%      each feature channel) depending on the value of the `Spatial`
%      option.
%
%   `Standardize``:: `false`
%      If `true`, use
%
%        (1/n sum_{q=1}^n X(i,j,q).^p + epsilon)^(1/p)
%
%      as normalisation factor. Combined with the `SubtractMean`
%      option, this can be used to divide the data by its sample
%      standard deviation.
%
%   See also: VL_NNNORMALIZE().

opts.epsilon = 1e-2 ;
opts.p = 2 ;
opts.spatial = false ;
opts.subtractMean = false ;
opts.standardize = false ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if opts.subtractMean
  if ~opts.spatial
    x = bsxfun(@minus, x, mean(x,3)) ;
  else
    x = bsxfun(@minus, x, mean(mean(x,1),2)) ;
  end
end

if ~opts.standardize
  if ~opts.spatial
    massp = sum(x.^opts.p,3) + opts.epsilon ;
  else
    massp = sum(sum(x.^opts.p,1),2) + opts.epsilon ;
  end
else
  if ~opts.spatial
    massp = mean(x.^opts.p,3) + opts.epsilon ;
  else
    massp = mean(mean(x.^opts.p,1),2) + opts.epsilon ;
  end
end
mass = massp.^(1/opts.p) ;
y = bsxfun(@rdivide, x, mass) ;

if nargin < 2 || isempty(dzdy)
  return ;
else
  dzdy = bsxfun(@rdivide, dzdy, mass) ;
  if ~opts.spatial
    tmp = sum(dzdy .* x, 3) ;
  else
    tmp = sum(sum(dzdy .* x, 1),2);
  end
  if opts.standardize
    if ~opts.spatial
      massp = massp * size(x,3) ;
    else
      massp = massp * size(x,1) * size(x,2) ;
    end
  end
  y = dzdy - bsxfun(@times, tmp, bsxfun(@rdivide, x.^(opts.p-1), massp)) ;
  if opts.subtractMean
    if ~opts.spatial
      y = bsxfun(@minus, y, mean(y,3)) ;
    else
      y = bsxfun(@minus, y, mean(mean(y,1),2)) ;
    end
  end
end
