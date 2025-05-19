gap_sizes_um = [8];
Forces=[7]

for i = 1:length(gap_sizes_um)
    for j= 1:length(Forces)
        run_single_sim(gap_sizes_um(i), gap_sizes_um, Forces(i));
    end
end
function run_single_sim(gap_size_um,gap_sizes_um, Force)
%cell transportation through a restricted thing
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Physical parameters
    %%-----------------------------------------------------------------------%%
    %% ----------------  Paper PHYSICAL INPUT  ---------------- %%
    phys.W_nm       = 200;        % nm corresponding to PF‑unit 1
    phys.Rpillar_um = 13.5;       % [µm]
    phys.Rcell_um   = 10;         % [µm]  (≈ radius from SI Fig S8)
    
    %% -----------------Paper -> Unitless! --------------------------------- %%
    psi_thresh = 0.8;
    conv   = 1000/phys.W_nm;      % nm ➜ PF units (==5)
    W      = 1;                   % PF
    dx     = 0.4*W;  dy = dx;     % dx/W = 0.4
    dt     = 1e-3;                % tune if stable
    nSteps = 100/dt;
    save_interval = round(.2/ dt);
    R_pillar = phys.Rpillar_um * conv; % 67.5
    R_cell   = phys.Rcell_um * conv;   % 50 PF‑units
    radius   = R_cell;                 % keep a short alias
    gap_size = gap_size_um*conv;
    
    %% --- domain intialization --- %%
    Lx = 3*(R_cell);   % PF‑units
    Ly = 9*R_cell;     % plenty vertical room
    
    Nx = ceil(Lx/dx);
    Ny = ceil(Ly/dy);
    
    x  = (0:Nx-1)*dx;
    y  = (0:Ny-1)*dy;
    [X,Y] = meshgrid(x,y);
    %%------ Values of paramiters ---- for $$\phi_t = \frac{\delta}{\delta \phi}U$$
    sigma = 3 * dx;  % term in front of \laplacian
    h = sigma; %"barrier" height
    v=Force*2; % how hard we pull
    lambda = 10; %strength of stopping flux
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%------ Saving-------
    gifFile = fullfile(getenv('HOME'), 'gifs', ...
        sprintf('trans_gap%d_Force_%d.gif', gap_size,v));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%--------Smooth tanh initial phase field (soft cell)------
    transition_width = 6; 
    cell_starting_point = .23; %where the middle of the cell starts as a %of domain
    cy_cell = round(cell_starting_point * Ny);
    cx_cell = round(Nx/2); %starts at half-way point of x-axis
    
    %defining cell (equation for cell)
    r = sqrt((X - x(cx_cell)).^2 + (Y - y(cy_cell)).^2);
    phi = 0.5 * (1 - tanh((r - radius) / transition_width));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Initiallizing the 2 pillars!
    % Center coordinates (index space)
    centerx = round(Nx/2);
    centery = round(Ny/2);
    % Parameters (PF)
    wall_thickness  = 1;       %how fast we go from 1 \to 0
    % Derived horizontal center-to-center distance
    gap_distance = 2 * R_pillar + gap_size;
    center_offset = gap_distance / 2;
    
    % Horizontal circle centers (in µm)
    x_left  = x(centerx) - center_offset;
    x_right = x(centerx) + center_offset;
    y_center = y(centery);
    x_center = x(centerx);
    
    % Distance fields from each center
    r_left  = sqrt((X - x_left).^2  + (Y- y_center).^2);
    r_right = sqrt((X - x_right).^2 + (Y - y_center).^2);
    
    % Smooth tanh profiles for each pillar
    psi_left  = 0.5 * (1 - tanh((r_left  - R_pillar) / wall_thickness));
    psi_right = 0.5 * (1 - tanh((r_right - R_pillar) / wall_thickness));
    
    % Combine into psi field
    psi = psi_left + psi_right;
    psi(psi > 1) = 1;  % clip to avoid values > 1
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Equations used:
    g= @(phi) phi.^3*(10 + 3*phi.*(2*phi-5));
    g_prime = @(phi) phi.^3.*(6*phi+3*(-5+2*phi))+3*phi.^2.*(10+3*phi.*(-5+2*phi));
    f = @(phi) 18*phi.^2.*(1- phi).^2;
    f_prime = @(phi) 36*(1 - phi).^2.*phi - 36*(1 - phi).*phi.^2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% used for finding velocity of our cell
    velocities = zeros(1, nSteps-1); % store velocity
    time_array = (0:nSteps-1) * dt;

    for step = 1:nSteps
        phi(psi>0.8) = 0; %makes sure that phi is not within our cell (there was some bleeding b4)
        g_prime_phi=g_prime(phi); %Used a lot throughout the loop
        lap_psi   = laplacian9(psi,dx,dy); %Used a lot throughout the loop
        % Mask φ *before* Laplacian
        phi_eff = phi;
        phi_eff(psi > psi_thresh) = 0;
        lap_phi = laplacian9(phi_eff, dx, dy);
        f_prime_phi = f_prime(phi)
        %%----Forces-----
            %% tension
            tension = 2*sigma*lap_phi - h*f_prime_phi
            % (optional belt‑and‑suspenders)
            tension(psi > psi_thresh) = 0; %stops bleeding
            %%----interaction between cell and pillar:
            interaction  = -lambda*lap_psi.*phi;
        %%---Frontal Force---
        % Apply mask in x-direction near the center
        % Find lowest y-position with phi > threshold
            %% creating our bands to sit around the head of the cell
            threshold = 0.5;
            phi_mask = phi > threshold;
            y_indices = find(any(phi_mask, 2));  % rows where phi > threshold
            if ~isempty(y_indices)
                bottom_idx = max(y_indices);
                y_bottom = y(bottom_idx);
            
                % Create band mask just above that region
                y_band = (Y > y_bottom - gap_size*0.5) & (Y < y_bottom + gap_size*0.5);
                x_band = abs(X - x_center) < gap_size / 4;
            
                mu = double(x_band & y_band);
            else
                mu = zeros(size(phi));  % fallback if phi is gone
            end 
        front = v * mu .* g_prime_phi; %frontal force used after finding band
        
        %%---Finding pressure term---
        F = tension + interaction + front;               
        % volume projection
        numerator   = sum(g_prime_phi.*F,'all');
        denominator = sum(g_prime_phi.^2,'all') + 1e-8;
        p = numerator / (h*denominator);
        %% Stepping in time!
        dphi_dt = F - p*h*g_prime_phi;
        phi     = phi + dt*dphi_dt;
        phi(phi>1)=1;  phi(phi<0)=0;% clip after update
        %% Getting our velocity:
        if step == 1
        y_com_prev = sum(Y(phi_mask)) / sum(phi_mask(:));
        end
        
        y_com = sum(Y(phi_mask)) / sum(phi_mask(:));
        velocities(step) = (y_com - y_com_prev) / dt;
        y_com_prev = y_com;
        %%----Plotting our the baddie moving!---
        if mod(step, save_interval) == 0 || step == 2
            both = phi + psi;
            %Plotting initial cell.
            % subplot(1,3,1); imagesc(g_prime_phi); title("g'(\phi)");
            % subplot(1,3,2); imagesc(mu); title("μ band");
            % subplot(1,3,3); imagesc(mu .* g_prime_phi); title("μ × g'");
            %% plotting function 
            fig = figure('Visible', 'on');
            tiledlayout(1,2, 'Padding', 'compact', 'TileSpacing', 'compact');
                %% -----Plot φ-----
                nexttile;
                imagesc(both, [0 1]); axis equal tight;
                colormap(spring); colorbar;
                title('\phi (Cell Shape)');
                %% ---- Plotting our velocity-----
                nexttile;
                velocity_plot = plot(NaN, NaN, 'LineWidth', 2); 
                xlabel('Time (s)'); ylabel('Velocity (\mum/s)');
                xlim([0, step*dt]);
                ylim([0, max(velocities)+1]);  % adjust this based on expected velocity range
                grid on;
                hold on;
                set(velocity_plot, 'XData', time_array(2:step), 'YData', velocities(1:step-1));
                drawnow limitrate;  % prevents lag by throttling redraw frequency
            %% Appending to our GIF!
            drawnow;
            frame = getframe(fig);
            [im, map] = rgb2ind(frame.cdata, 256);

            if step == 2                            % first time → create file
                imwrite(im, map, gifFile, 'gif', ...
                    'LoopCount', inf, 'DelayTime', 0);  % DelayTime ~ seconds per frame
            else                                    % later → append
                imwrite(im, map, gifFile, 'gif', ...
                    'WriteMode', 'append', 'DelayTime', 0);
            end
        end
    
    
    
    end
    
    
    function lap = laplacian9(phi, dx, dy)
        kernel =[1 4 1; 4 -20 4; 1 4 1] / 6;
        lap = conv2(phi, kernel, 'same') / dx^2;
    end
    function [dphidx, dphidy] = my_gradient(phi, dx, dy)
    %MY_GRADIENT Fast 2D gradient for uniform grid spacing
    %   [dphidx, dphidy] = my_gradient(phi, dx, dy)
    %
    %   Inputs:
    %     phi - 2D array
    %     dx  - spacing in x-direction (columns)
    %     dy  - spacing in y-direction (rows)
    %
    %   Outputs:
    %     dphidx - partial derivative ∂φ/∂x
    %     dphidy - partial derivative ∂φ/∂y
    
        if nargin < 2
            dx = 1;
        end
        if nargin < 3
            dy = 1;
        end
    
        [Ny, Nx] = size(phi);
    
        dphidx = zeros(Ny, Nx);
        dphidy = zeros(Ny, Nx);
    
        % ∂φ/∂x: finite difference across columns
        dphidx(:,2:Nx-1) = (phi(:,3:Nx) - phi(:,1:Nx-2)) / (2*dx);
        dphidx(:,1)      = (phi(:,2) - phi(:,1)) / dx;
        dphidx(:,end)    = (phi(:,end) - phi(:,end-1)) / dx;
    
        % ∂φ/∂y: finite difference across rows
        dphidy(2:Ny-1,:) = (phi(3:Ny,:) - phi(1:Ny-2,:)) / (2*dy);
        dphidy(1,:)      = (phi(2,:) - phi(1,:)) / dy;
        dphidy(end,:)    = (phi(end,:) - phi(end-1,:)) / dy;
    end
    function div = my_divergence(vx, vy, dx, dy)
    %MY_DIVERGENCE Fast 2D divergence for uniform grid spacing
    %   div = my_divergence(vx, vy, dx, dy)
    %
    %   Inputs:
    %     vx - x-component of vector field (same size as vy)
    %     vy - y-component of vector field
    %     dx - spacing in x-direction
    %     dy - spacing in y-direction
    %
    %   Output:
    %     div - scalar divergence field ∇·v = ∂vx/∂x + ∂vy/∂y
    
        if nargin < 3
            dx = 1;
        end
        if nargin < 4
            dy = 1;
        end
    
        [Ny, Nx] = size(vx);
        div = zeros(Ny, Nx);
    
        % ∂vx/∂x
        dvxdx = zeros(Ny, Nx);
        dvxdx(:,2:Nx-1) = (vx(:,3:Nx) - vx(:,1:Nx-2)) / (2*dx);
        dvxdx(:,1)      = (vx(:,2) - vx(:,1)) / dx;
        dvxdx(:,end)    = (vx(:,end) - vx(:,end-1)) / dx;
    
        % ∂vy/∂y
        dvydy = zeros(Ny, Nx);
        dvydy(2:Ny-1,:) = (vy(3:Ny,:) - vy(1:Ny-2,:)) / (2*dy);
        dvydy(1,:)      = (vy(2,:) - vy(1,:)) / dy;
        dvydy(end,:)    = (vy(end,:) - vy(end-1,:)) / dy;
    
        % Total divergence
        div = dvxdx + dvydy;
    end
end