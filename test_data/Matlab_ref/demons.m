% Demons Registration
%
% Simple matlab code for 3D image registration using the diffeomorphic log-demons algorithm 
% Code is provided in order to help the understanding of the Demons algorithm
%
% Original Algorithm in:
% [1] - Symmetric Log-Domain Diffeomorphic Registration: A Demons-Based Approach
%       Vercauteren, Pennec, Perchant, Ayache -- MICCAI 2008, 754-761
% [2] - Diffeomorphic demons: Efficient non-parametric image registration,
%       Vercauteren, Pennec, Perchant, Ayache -- NeuroImage 2009, (45)1:61-72
%
% For a more recent work/survey, exploiting global shape characteristics 
% (instead of the conventional local gradient-based approaches), consider citing 
%
% [1] - Spectral Log-Demons: Diffeomorphic Image Registration with Very Large Deformations
%       Lombaert, Grady, Pennec, Ayache, Cheriet -- IJCV 2014, (107)3:254-271

addpath Matlab_IO
addpath Matlab_ref
addpath test_data


figure(1); clf; colormap gray;

%% Parameters
niter           = 10;
sigma_fluid     = 3.0; % regularize update      field
sigma_diffusion = 2.0; % regularize deformation field
sigma_i         = 1.0; % weight on similarity term
sigma_x         = 1.0; % weight on spatial uncertainties (maximal step)
diffeomorphic   = 1;   % use exp(u)
nlevel          = 1;   % multiresolution
do_display      = 1;   % display iterations

% Load images
F=loadbin('fixedBrain_1.dat');
M=loadbin('movingBrain_1.dat');
F=255*(F-min(F(:)))/range(F(:));
M=255*(M-min(M(:)))/range(M(:));

if nlevel == 1
    
    %% Register
    disp(['Register...']);
    opt = struct('niter',niter, ...
        'sigma_fluid',sigma_fluid, ...
        'sigma_diffusion',sigma_diffusion, ...
        'sigma_i',sigma_i, ...
        'sigma_x',sigma_x, ...
        'diffeomorphic',diffeomorphic, ...
        'do_display',do_display, ...
        'do_plotenergy',1);
    
    [Mp,sx,sy,sz] = register(F,M,opt);

else
    
    %% Multiresolution
    sx = zeros(size(M)); % deformation field
    sy = zeros(size(M));
    sz = zeros(size(M));
    for k=nlevel:-1:1
        disp(['Register level: ' num2str(k) '...']);

        % downsample
        scale = 2^-(k-1);
        Fl  = resize(F,scale);
        Ml  = resize(M,scale);
        sxl = resize(sx*scale,scale);
        syl = resize(sy*scale,scale);
        szl = resize(sz*scale,scale);

        % register
        opt = struct('niter',niter,...
                     'sigma_fluid',sigma_fluid,...
                     'sigma_diffusion',sigma_diffusion,...
                     'sigma_i',sigma_i,...
                     'sigma_x',sigma_x,...
                     'diffeomorphic',diffeomorphic,...
                     'vx',sxl, 'vy',syl, 'vz',szl,...
                     'do_display',do_display, 'do_plotenergy',1);
        [Mp,sxl,syl,szl] = register(Fl,Ml,opt);

        % upsample
        sx = resize(sxl/scale,size(M));
        sy = resize(syl/scale,size(M));
        sz = resize(szl/scale,size(M));
    end
    
end

