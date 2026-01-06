%% Trajectory Generator

% Time vector
T = 5;
t = linspace(0, T, 1001);

q0 = [0 pi/2];
qf = [pi/2 pi/2];

% Joint trajectory
[q, qd, qdd] = jtraj(q0, qf, t);

% fix shape with transpose:
if size(q,1) ~= 2
    q   = q';
    qd  = qd';
    qdd = qdd';
end

% Preallocate
N = length(t);
xd  = zeros(2, N);
xdot_d = zeros(2, N);
xdd_d  = zeros(2, N);

function J = jacobian(q)
    % Jacobian

    q1 = q(1);        q2 = q(2);
    J11 = -sin(q1) - sin(q1 + q2);
    J12 = -sin(q1 + q2);
    J21 =  cos(q1) + cos(q1 + q2);
    J22 =  cos(q1 + q2);
    
    J = [J11 J12;
         J21 J22];
end

function Jdot = jacobianDot(q, qdot)
    % Derivative of Jacobian
    
    q1 = q(1);        q2 = q(2);
    dq1 = qdot(1);    dq2 = qdot(2);
    
    Jdot11 = -cos(q1)*dq1 - cos(q1 + q2)*(dq1 + dq2);
    Jdot12 = -cos(q1 + q2)*(dq1 + dq2);
    
    Jdot21 = -sin(q1)*dq1 - sin(q1 + q2)*(dq1 + dq2);
    Jdot22 = -sin(q1 + q2)*(dq1 + dq2);
    
    Jdot = [Jdot11 Jdot12;
            Jdot21 Jdot22];
end

function x = fk(q)

    q1 = q(1);
    q2 = q(2);
    
    xpos = cos(q1) + cos(q1 + q2);
    ypos = sin(q1) + sin(q1 + q2);
    
    x = [xpos; ypos];
end


for k = 1:N
   J = jacobian(q(:,k));
   Jdot = jacobianDot(q(:,k), qd(:,k));

   xd(:,k)      = fk(q(:,k));
   xdot_d(:,k)  = J * qd(:,k);
   xdd_d(:,k)   = J * qdd(:,k) + Jdot * qd(:,k);
end

% Store for Simulink
xd_sim = [time_sim, xd'];     % N×3
xdot_sim = [time_sim, xdot_d'];  
xddot_sim = [time_sim, xdd_d'];
