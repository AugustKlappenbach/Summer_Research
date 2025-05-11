% Full Phase Field Model with Reaction-Diffusion Coupling
% Parameters from the table
Nx = 200; Ny = 250;
dx = 0.1; dy = 0.1;
%dt = 1e-4/8; Old dt. For bending



dt = 1e-3;
nSteps = 48000000;
% Define the full output path
output_dir = 'C:/Users/zackk/OneDrive/Desktop/Research/With bending results';
save_interval = round(2 / dt); 

% Make sure the directory exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end



% Physical parameters
gamma = 1.0;            % pN
kappa = 1.0;            % pN*um^2
alpha = 0.3;            % pN/um
beta = 0.5;             % pN/um
Ma = 4.0;               % pN/um^3
A0 = 133.5;             % um^2
epsilon = 1.0;          % um
tau = 2.62;             % pN*s/um^2

% Reaction-Diffusion constants
a = 0.084;              % s^-1
b = 1.146;              % s^-1
c = 0.0764;             % s^-1
e = 0.107;              % s^-1
DV = 0.382;             % um^2/s
DW = 0.0764;            % um^2/s
lambda_wall = 10;  % strength of penalty



% Spatial grid
x = linspace(0, (Nx-1)*dx, Nx);
y = linspace(0, (Ny-1)*dy, Ny);
[X, Y] = meshgrid(x, y);

% Initial phase field: circle of radius 4 um
phi = zeros(Ny, Nx);
cx = round(Nx/2);              % center in x
cy = round(0.15 * Ny);        
radius = 6.52;                  
phi(((X - x(cx)).^2 + (Y - y(cy)).^2) < radius^2) = 1;

phi_prev=phi;
phi = phi + 0.01 * randn(size(phi));
phi(phi < 0) = 0;
phi(phi > 1) = 1;
phi_old = phi;

psi = zeros(Ny, Nx);

% Parameters (all in microns)
circle_radius = 3.0;       % radius of each wall circle
gap_size = 3.0;            % µm gap between circles
wall_thickness = 0.5;      % width of tanh transition zone

% Center coordinates (index space)
cx = round(Nx/2);
cy = round(Ny/2);

% Derived vertical offsets
gap_distance = 2 * circle_radius + gap_size;
center_offset = gap_distance / 2;

% Convert X and Y to µm coordinates (if not already)
[X_um, Y_um] = meshgrid(x, y);
% Parameters (µm)
circle_radius   = 3.0;       % µm
gap_size        = 3.0;       % µm between circle edges
wall_thickness  = 0.5;       % µm tanh transition zone

% Center coordinates (index space)
cy = round(Ny / 2);
cx = round(Nx / 2);

% Derived horizontal center-to-center distance
gap_distance = 2 * circle_radius + gap_size;
center_offset = gap_distance / 2;

% Convert to µm coordinates
[X_um, Y_um] = meshgrid(x, y);

% Horizontal circle centers (in µm)
x_left  = x(cx) - center_offset;
x_right = x(cx) + center_offset;
y_center = y(cy);

% Distance fields from each center
r_left  = sqrt((X_um - x_left).^2  + (Y_um - y_center).^2);
r_right = sqrt((X_um - x_right).^2 + (Y_um - y_center).^2);

% Smooth tanh profiles for each wall circle
psi_left  = 0.5 * (1 - tanh((r_left  - circle_radius) / wall_thickness));
psi_right = 0.5 * (1 - tanh((r_right - circle_radius) / wall_thickness));

% Combine into psi field
psi = psi_left + psi_right;
psi(psi > 1) = 1;  % clip to avoid values > 1


figure; imagesc(psi); axis equal tight; colorbar;
title('\psi (tanh walls)');


% Initialize V and W inside the cell only
V = 1.1 * phi;
bias = 0.4;
V = V + bias * phi .* (Y > y(cy));  % adds rear-to-front polarity


W = zeros(Ny, Nx);  % (200 x 600)
W0 = 30;

% Create the rear-loaded bundle shape
rear_shape = W0 * exp(-((Y - y(cy) + 5).^2) / (2 * 3^2));

% Apply to rear side of the cell only
W = rear_shape .* phi .* (Y < y(cy));
%GPU arrays:
% phi = gpuArray(phi);
% V = gpuArray(V);
% W = gpuArray(W);
% X = gpuArray(X);
% Y = gpuArray(Y);
% phi_prev = gpuArray(phi_prev);

% Custom Laplacian function
function lap = laplacian9(phi, dx, dy)
    %kernel = gpuArray([1 4 1; 4 -20 4; 1 4 1]) / 6;
    kernel =[1 4 1; 4 -20 4; 1 4 1] / 6;
    lap = conv2(phi, kernel, 'same') / dx^2;
end



G_prime = @(phi) 36 * phi .* (1 - phi) .* (1 - 2 * phi);
G_double_prime = @(phi) 36 * ((1 - phi) .* (1 - 2*phi) - phi .* (1 - 2*phi) - 2 * phi .* (1 - phi));

% Tracking center of mass and area
xcom = zeros(1, nSteps);
ycom = zeros(1, nSteps);
area_vals = zeros(1, nSteps);
fig = figure('Visible', 'on');
    imagesc(phi, [0 1]); axis equal tight; hold on;
    contour(psi, [0.5 0.5], 'k', 'LineWidth', 2);
    title('\phi with wall overlay'); colorbar;

for step = 1:nSteps
    phi_old = phi;
    % Tension
    lap_phi = laplacian9(phi, dx, dy);
    tension = gamma * (lap_phi - G_prime(phi) / epsilon^2);

    % Volume conservation
    A = sum(phi(:)) * dx * dy;
    phi_smooth = imgaussfilt(phi, 1);
    [dphix, dphiy] = gradient(phi_smooth, dx, dy);
    grad_phi_mag = sqrt(dphix.^2 + dphiy.^2);
    volume = -Ma * (A - A0) * grad_phi_mag;

    % % Bending 
    % biharm_phi = laplacian9(lap_phi, dx, dy);
    % lap_Gp = laplacian9(G_prime(phi), dx, dy);
    % Gpp = G_double_prime(phi);
    % bending = -kappa * (biharm_phi - lap_Gp - Gpp .* lap_phi + Gpp .* G_prime(phi));

    % Reaction-Diffusion term
    reaction_diffusion = (alpha * V ) .* grad_phi_mag;
    %Penalty term:
    penalty = -lambda_wall * psi; 
    % Update phi
    %dphi_dt = (tension + volume + reaction_diffusion + penalty + bending) / tau;
    dphi_dt = (tension + volume + reaction_diffusion + penalty ) / tau;
    phi = phi + dt * dphi_dt;
    phi(phi < 0) = 0;
    phi(phi > 1) = 1;


    % Update V and W
    RV = phi_old .* (a - b * V .* W.^2 - c * V);
    RW = phi_old .* (b * V .* W.^2 - e * W);

    [phix, phiy] = gradient(phi_old, dx, dy);
    [Vx, Vy] = gradient(V, dx, dy);
    [Wx, Wy] = gradient(W, dx, dy);
    
    phiVx = phix .* V + phi_old .* Vx;
    phiVy = phiy .* V + phi_old .* Vy;
    phiWx = phix .* W + phi_old .* Wx;
    phiWy = phiy .* W + phi_old .* Wy;
    
    div_phi_gradV = divergence(phiVx, phiVy);
    div_phi_gradW = divergence(phiWx, phiWy);

    V = V + dt ./(phi_old + 1e-8) .* (RV + DV * div_phi_gradV);
    W = W + dt ./ (phi_old + 1e-8) .* (RW + DW * div_phi_gradW);

    V(phi_old < 1e-3) = 0;
    W(phi_old < 1e-3) = 0;

    % Track center of mass and area
    total_phi = sum(phi(:));
    xcom(step) = sum(sum(phi_old .* X)) / total_phi;
    ycom(step) = sum(sum(phi_old .* Y)) / total_phi;
    area_vals(step) = A;
    % if mod(step, 5000) == 0
    % fprintf("RD max: %.4f | mean: %.4f\n Area value: %.4f\n", max(reaction_diffusion(:)), mean(reaction_diffusion(:)),A);
    % end
    % if mod(step, 10000) == 0
    % delta_phi = abs(phi - phi_prev);
    % fprintf("Mean Δφ: %.6f | Max Δφ: %.6f\n", mean(delta_phi(:)), max(delta_phi(:)));
    % phi_prev = phi;
    % phi_mask = phi > 0.5;
    % [y_indices, x_indices] = find(phi_mask);
    % width = range(x_indices) * dx;
    % height = range(y_indices) * dy;
    % fprintf("Shape | width: %.2f µm, height: %.2f µm\n", width, height);
    % elongation = height / width;
    % fprintf("Elongation ratio: %.3f\n", elongation);
    % end6+
    %40000 
 if mod(step, save_interval) == 0
    fig = figure('Visible', 'on');
    imagesc(phi, [0 1]); axis equal tight; hold on;
    
    % Overlay the wall as transparent mask
    h = imagesc(psi);
    set(h, 'AlphaData', 0.3 * (psi > 0));  % semi-transparent wall only where psi > 0
    colormap('parula');
    title('\phi with wall overlay'); colorbar;
    drawnow;
end


end
