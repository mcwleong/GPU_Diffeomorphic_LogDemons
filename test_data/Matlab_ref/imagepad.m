%% Pad image
%  Changed: Dec 31st, 2011
%
function [I,lim] = imagepad(I,scale)

    if nargin<2; scale = 2; end; % default, pad image twice as big
    
    if size(size(I),2)>3; [I,lim]=imagepad_multichannel(I,scale); return; end; % check if I is multichannel
    
    Ip  = zeros(ceil(size(I)*scale));
    lim = bsxfun(@plus, floor(size(I)*(scale-1)/2), [[1 1 1]; size(I)]); % image limits
    Ip(lim(1):lim(2),lim(3):lim(4),lim(5):lim(6)) = I;                   % pad image
    I = Ip;

end

%% Pad image
%  Changed: Dec 31st, 2011
%
function [I,lim] = imagepad_multichannel(I,scale)

    if nargin<2; scale = 2; end; % default, pad image twice as big
    nchannels = size(I,4);

    for i=1:nchannels
        Ip(:,:,:,i) = imagepad(I(:,:,:,i),scale);
    end
        
    imagesize = [size(I,1) size(I,2) size(I,3)];
    lim = bsxfun(@plus, floor(imagesize*(scale-1)/2), [[1 1 1]; imagesize]); % image limits
    I = Ip;

end

