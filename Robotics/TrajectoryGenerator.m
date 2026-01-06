%% Trajectory Generator

% Time vector
T = 5;
numPeriods = 5;
T_total = T*numPeriods;
t = linspace(0, T_total, 1001);
N = length(t);

% Define endpoints in Cartesian space
A = [1.5; 0.5];
B = [1.2; 1.2];
D = B - A;             % Direction vector between points

% Preallocate trajectory arrays
xd      = zeros(2, N);
xdot_d  = zeros(2, N);
xdd_d   = zeros(2, N);

% Sine interpolation functions
omega = 2*pi/T;

for k = 1:N
    tk = t(k);

    % Smooth cosine-based back-and-forth interpolation
    s      = (1 - cos(omega*tk)) / 2;
    sdot   = (omega/2) * sin(omega*tk);
    sddot  = (omega^2/2) * cos(omega*tk);

    % Desired position, velocity, acceleration
    xd(:,k)      = A + s * D;
    xdot_d(:,k)  = sdot  * D;
    xdd_d(:,k)   = sddot * D;
end

% Store for Simulink (assuming time_sim = t(:))
time_sim = t(:);
xd_sim     = [time_sim, xd'];
xdot_sim   = [time_sim, xdot_d'];
xddot_sim  = [time_sim, xdd_d'];
