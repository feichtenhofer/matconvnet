function a = vl_accumulate(alpha, a, beta, b)
%VL_ACCUMULATE  Compute alpha A + beta B
%   C = VL_ACCUMULATE(ALPHA, A, BETA, B) computes
%   C = alpha A + beta B.

if isscalar(a)
  a = alpha * a + beta * b ;
  return ;
elseif isa(a, 'gpuArray')
  vl_accumulatemex(alpha, a, beta, b, 'inplace') ;
else
  a = vl_accumulatemex(alpha, a, beta, b) ;
end
