%%% Jack Landers - Robotics

%%% Manipulator Dynamics

%% 1. EOM for a polar two-link manipulator
clc;
syms g q1 d2 L1

R01 = [cos(q1) -sin(q1) 0;
       sin(q1) cos(q1) 0
       0 0 1];
P01 = [0; 0; L1];

R12 = [1 0 0;
       0 0 1;
       0 -1 0];
P12 = [0; d2; 0];

P1c1 = [0; 0; 0];   % COM at frame 1 origin
P2c2 = [0; 0; -d2];

f33 = [0; 0; 0];
n33 = f33;
v00 = f33;
w00 = f33;
wdot00 = f33;
vdot00 = [0; 0; g]; % Gravity

% Outward Link 1
syms qdot1 qdotdot1 m1

w11 = R01'*w00 + [0; 0; qdot1];
wdot11 = R01'*w00 + cross(R01'*w00, [0; 0; qdot1]) + [0; 0; qdotdot1];

vdot11 = R01'*(cross(w00, P01) + cross(w00, cross(w00, P01)) + vdot00);
vdot1c1 = cross(wdot11, P1c1) + cross(w11, cross(w11, P1c1)) + vdot11;

F11 = m1*vdot1c1;
N11 = 0;

% Outward Link 2
syms ddot ddotdot m2

w22 = R12'*w11;
wdot22 = R12'*wdot11 + cross(R12'*w11, [0; 0; 0]) + [0; 0; 0];

vdot22 = R12'*(cross(wdot11, P12) + cross(w11, cross(w11, P12)) + vdot11) + 2*cross(w22, [0; 0; ddot]) + [0;0;ddotdot];
vdot2c2 = cross(wdot22, P2c2) + cross(w22, cross(w22, P2c2)) + vdot22;

F22 = m2*vdot2c2;
N22 = 0;

% Inward Link 2

f22 = F22
n22 = cross(P2c2, F22);

% Inward Link 1

f11 = R12*f22 + F11;
n11 = N11 + R01'*n22 + cross(P1c1, F11) + cross(P12, R12*f22)

% Extract Actuator Force/Torques

T2 = f22(3)
T1 = n11(3)


%% Cartesian Space EOM

% T2 contains the M part
% T1 contains the V part
% Jacobian w/ Velocity Propagation

syms w1 a2;

v0 = [0; 0; 0;];
w0 = [0; 0; 0;];

w11 = R01'*w0+[0; 0; w1];
v11 = R01'*(v0+cross(w0,P01));

w22 = R12'*w11;
v22 = R12'*(v11+cross(w11,P12))+[0; 0; a2];

J = jacobian(v22, [w1; a2;])

Jq1 = diff(J, q1);
Jq2 = diff(J, d2);
Jdot = Jq1*qdot1 + Jq2*ddot;

M = [0 0; 0 m2];
V = [2*d2*ddot*m2*qdot1; 0];

Mx = pinv(J)'*M*pinv(J)

Vx = pinv(J)'*(V-M*pinv(J)*Jdot*[qdot1; ddot])
