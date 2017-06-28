function removeLayer(obj, name, chainInputs)
%REMOVELAYER Remove a layer from the network
%   REMOVELAYER(OBJ, NAME) removes the layer NAME from the DagNN object
%   OBJ. NAME can be a string or a cell array of strings.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if ischar(name), name = {name}; end;
f = obj.getLayerIndex(name);
if any(isnan(f))
  error('Invalid layer name `%s`', ...
    strjoin(name(isnan(idxs)), ', '));
end
if nargin < 3, chainInputs = false; end


if chainInputs
  for f_l = f
    % chain input of l that has layer as input
    for l = 1:numel(obj.layers)    
      for i = obj.layers(f_l).outputIndexes
        sel = find(intersect(obj.layers(l).inputIndexes, i));
        if any(sel)
          obj.layers(l).inputs{sel} = obj.layers(f_l).inputs{1};
        end
      end
    end
  end
end
obj.layers(f) = [] ;
obj.rebuild() ;
