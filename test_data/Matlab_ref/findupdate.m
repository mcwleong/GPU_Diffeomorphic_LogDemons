%% Find update between two images
%  Changed: Dec 31st, 2011
%

function [ux,uy,uz, gx, gy, gz, sgn] = findupdate(F,Mp,sigma_i,sigma_x)


    % image difference
    diff = F - Mp;
    
    % fixed image gradient
    [gx_f,gy_f,gz_f] = gradient(F);          % image gradient
    gx_f(1,:,:) =0;gx_f(end,:,:) =0; gx_f(:,1,:)=0; gx_f(:,end,:)=0; gx_f(:,:,1)=0; gx_f(:,:,end)=0;
    gy_f(1,:,:) =0;gy_f(end,:,:) =0; gy_f(:,1,:)=0; gy_f(:,end,:)=0; gy_f(:,:,1)=0; gy_f(:,:,end)=0;
    gz_f(1,:,:) =0;gz_f(end,:,:) =0; gz_f(:,1,:)=0; gz_f(:,end,:)=0; gz_f(:,:,1)=0; gz_f(:,:,end)=0;
    normg2_f  = gx_f.^2 + gy_f.^2 + gz_f.^2; % squared norm of gradient
    
    % moving image gradient
    [gx,gy,gz] = gradient(Mp);          % image gradient
    gx(1,:,:) =0;gx(end,:,:) =0; gx(:,1,:)=0; gx(:,end,:)=0; gx(:,:,1)=0; gx(:,:,end)=0;
    gy(1,:,:) =0;gy(end,:,:) =0; gy(:,1,:)=0; gy(:,end,:)=0; gy(:,:,1)=0; gy(:,:,end)=0;
    gz(1,:,:) =0;gz(end,:,:) =0; gz(:,1,:)=0; gz(:,end,:)=0; gz(:,:,1)=0; gz(:,:,end)=0;
    
    normg2  = gx.^2 + gy.^2 + gz.^2;         % squared norm of gradient
    
    % update is Idiff / (||J||^2+(Idiff^2)/sigma_x^2) J, with Idiff = F(x)-M(x+s), and J = Grad(M(x+s));
    scale = diff ./ (normg2 + diff.^2*sigma_i^2/sigma_x^2);
    scale(normg2==0) = 0;
    scale(diff  ==0) = 0;
    scale = scale .* ((scale>=0) + (scale<0) .* sign(gx.*gx_f + gy.*gy_f + gz.*gz_f)); % avoid collapsing gradients (change sign if moving goes backward)
    sgn = sign(gx.*gx_f + gy.*gy_f + gz.*gz_f);
    ux = gx .* scale;
    uy = gy .* scale;
    uz = gz .* scale;
    
    % Zero non overlapping areas
    %ux(F==0)       = 0; uy(F==0)       = 0;
    %ux(M_prime==0) = 0; uy(M_prime==0) = 0;

end
