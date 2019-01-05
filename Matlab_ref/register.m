%% Register two images
%  Changed: Dec 31st, 2011
%
function [Mp,sx,sy,sz] = register(F,M,opt)


if nargin<3;  opt = struct();  end;
if ~isfield(opt,'sigma_fluid');      opt.sigma_fluid     = 3.0;              end;
if ~isfield(opt,'sigma_diffusion');  opt.sigma_diffusion = 2.0;              end;
if ~isfield(opt,'sigma_i');          opt.sigma_i         = 1.0;              end;
if ~isfield(opt,'sigma_x');          opt.sigma_x         = 1.0;              end;
if ~isfield(opt,'niter');            opt.niter           = 250;              end;
if ~isfield(opt,'sx');               opt.sx              = zeros(size(M));   end;
if ~isfield(opt,'sy');               opt.sy              = zeros(size(M));   end;
if ~isfield(opt,'sz');               opt.sz              = zeros(size(M));   end;
if ~isfield(opt,'imagepad');         opt.imagepad        = 1.2;              end;
if ~isfield(opt,'stop_criterium');   opt.stop_criterium  = 1e-4;             end;
if ~isfield(opt,'do_display');       opt.do_display      = 1;                end;
if ~isfield(opt,'do_plotenergy');    opt.do_plotenergy   = 1;                end;
if ~isfield(opt,'do_avi');           opt.do_avi          = 0;                end;

if opt.do_avi
    if ~isfield(opt, 'aviobj')
        opt.aviobj = avifile('spectral-demons.avi','compression','None', 'fps',10);
        opt.do_closeavi = 1; % create and close avi file here
    end
    if ~isfield(opt, 'do_closeavi'); opt.do_closeavi = 0; end;
    global aviobj;
    aviobj = opt.aviobj;
end;

%% Pad image
[F,lim] = imagepad(F,opt.imagepad);
[M,lim] = imagepad(M,opt.imagepad);

%% T is the deformation from F to M
if ~isempty(opt.sx) && ~isempty(opt.sy) && ~isempty(opt.sz)
    sx = imagepad(opt.sx,opt.imagepad);
    sy = imagepad(opt.sy,opt.imagepad);
    sz = imagepad(opt.sz,opt.imagepad);
end
vx = zeros(size(M));
vy = vx;
vz=vy;
e  = zeros(1,opt.niter);
e_min = 1e+10;      % Minimal energy

Mp = M;
%% Iterate update fields
%for iter=1:opt.niter
for iter=1:opt.niter
    % Find update
    [ux,uy,uz] = findupdate(F,Mp,opt.sigma_i,opt.sigma_x);
    % Regularize update
    ux    = imgaussian(ux,opt.sigma_fluid);
    uy    = imgaussian(uy,opt.sigma_fluid);
    uz    = imgaussian(uz,opt.sigma_fluid);
      
    % Update correspondence (demons) - composition
    [vx,vy,vz] = compose(vx,vy,vz,ux, uy, uz);
    
    % regularize velocity field
    vx = imgaussian(vx,opt.sigma_diffusion);
    vy = imgaussian(vy,opt.sigma_diffusion);
    vz = imgaussian(vz,opt.sigma_diffusion);
    
    % Get Transformation
    [sx,sy,sz] = expfield(vx,vy,vz);  % deformation field
    Mp = iminterpolate(M, sx, sy, sz);
    % Compute energy
    e(iter) = energy(F,Mp,sx,sy,sz,opt.sigma_i,opt.sigma_x);
    disp(['Iteration: ' num2str(iter) ' - ' 'energy: ' num2str(e(iter))]);
    if e(iter)<e_min
        sx_min = sx; sy_min = sy; sz_min = sz;
        e_min  = e(iter);
    end
    
    % Stop criterium
    %         if iter>1 && abs(e(iter) - e(max(1,iter-5))) < e(1)*opt.stop_criterium
    %             break;
    %         end
    
    if opt.do_display
        % display deformation
        subplot(2,4,7); showvector(ux,uy,uz,4,0,lim); title('Update');
        subplot(2,4,8); showgrid  (sx,sy,sz,4,lim);   title('Transformation');
        drawnow;
        
        % Display registration
        diff   = (F-Mp).^2;
        showimage(F,'Fixed', M,'Moving', Mp,'Warped', diff,'Diff', 'lim',lim,'nbrows',2,'caxis',[0 256]); drawnow;
        
        % Plot energy
        if opt.do_plotenergy
            subplot(2,2,3)
            hold on;
            plot(1:iter,e(1:iter),'r-'); xlim([0 opt.niter]);
            xlabel('Iteration'); ylabel('Energy');
            hold off;
            drawnow
        end
    end
    
    if opt.do_avi; aviobj = addframe(aviobj,getframe(gcf)); end;
    saveToBin(Mp, ['C:\\Users\\Martin\\Documents\\gpu_diffeomorphic_logdemons_private\\test_data\\register_results\\matlab_results\\Mpm_' num2str(iter) '.bin']);
end


%% Get Best Transformation
sx = sx_min;  sy = sy_min;  sz = sz_min;

%% Transform moving image
Mp = iminterpolate(M,sx,sy,sz);

%% Unpad image
Mp = Mp(lim(1):lim(2),lim(3):lim(4),lim(5):lim(6));

sx = sx(lim(1):lim(2),lim(3):lim(4),lim(5):lim(6));
sy = sy(lim(1):lim(2),lim(3):lim(4),lim(5):lim(6));
sz = sz(lim(1):lim(2),lim(3):lim(4),lim(5):lim(6));

if opt.do_avi && opt.do_closeavi
    aviobj = close(aviobj);
end

end

%% Get energy
function e = energy(F,Mp,sx,sy,sz, sigma_i,sigma_x)

% Intensity difference
diff2  = (F-Mp).^2;
area   = numel(Mp);

% Transformation Gradient
jac = jacobian(sx,sy,sz);

% Three energy components
e_sim  = sum(diff2(:)) / area;
e_reg  = sum(jac(:).^2) / area;

% Total energy
e      = e_sim + (sigma_i^2/sigma_x^2) * e_reg;

end

