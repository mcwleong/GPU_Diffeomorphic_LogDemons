%% Apply gaussian filter to image
%  Changed: Dec 31st, 2011
%
function [Iz] = imgaussian(I,sigma)

    if sigma==0; return; end; % no smoothing
    radius = ceil(3*sigma);
hx = -radius:radius;
hx = exp(-hx .^ 2 / (2 * sigma ^ 2));
hx = hx / sum (hx);
hy = reshape(hx,[],1);
hz = reshape(hx, 1,1,[]);
h2 = convn( hx, hy);
h3 = convn( h2, hz);
tic
Ix= convn(I, hx,'same');
Iy = convn(Ix, hy,'same');
Iz = convn(Iy, hz, 'same');
    
end

%% Apply gaussian filter to image
%  Changed: Dec 31st, 2011
%
function I = imgaussian_multichannel(I,sigma)

    if sigma==0; return; end; % no smoothing
    
    nchannels = size(I,4);

    for i=1:nchannels
        Ip(:,:,:,i) = imgaussian(I(:,:,:,i),sigma);
    end
    
    I = Ip;

end

