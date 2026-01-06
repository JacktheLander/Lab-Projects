%%% Jack Landers - Robotics

%%% Homework 4
%% 6. Direct Differentiation

clearvars;
syms theta1 theta2 theta3 L1 L2 L3

c1 = cos(theta1); s1 = sin(theta1);
c2 = cos(theta2); s2 = sin(theta2);
c3 = cos(theta3); s3 = sin(theta3);

T01 = [c1 -s1 0 0;
       s1 c1 0 0;
       0 0 1 0;
       0 0 0 1];

T12 = [c2 -s2 0 L1;
       0 0 -1 0;
       s2 c2 0 0;
       0 0 0 1];

T23 = [c3 -s3 0 L2;
       s3 c3 0 0;
       0 0 1 0;
       0 0 0 1];

T3ee = [1 0 0 L3;
        0 1 0 0;
        0 0 1 0;
        0 0 0 1];

T02 = T01*T12;
T03 = T02*T23;
T0ee = simplify(T03*T3ee)

X = T0ee(1:3,4)
J = [diff(X, theta1), diff(X, theta2), diff(X, theta3)];
J = simplify(J)

%% Velocity Propagation

syms w1 w2 w3;

R01 = T01(1:3,1:3);
R12 = T12(1:3,1:3);
R23 = T23(1:3,1:3);
R3ee = T3ee(1:3,1:3);

P01 = T01(1:3,4);
P12 = T12(1:3,4);
P23 = T23(1:3,4);
P3ee = T3ee(1:3,4);


v0 = [0; 0; 0;];
w0 = [0; 0; 0;];

w11 = R01*w0+[0; 0; w1]
v11 = R01*(v0+cross(w0,P01))

w22 = R12*w11+[0; 0; w2]
v22 = R12*(v11+cross(w11,P12))

w33 = R23*w22+[0; 0; w3]
v33 = R23*(v22+cross(w22,P23))

wee = R3ee*w33+[0;0;0]
vee = R3ee*(v33+cross(w33,P3ee))

J = jacobian(vee, [w1; w2; w3])

% I used the wrong rotation matrices. It did not make intuitive sense to
% me why we use the rotation from the next frame to the current when
% propagating forwards

%% Static Force Propagation

syms fx fy fz;

fee = [fx; fy; fz];
nee = [0; 0; 0];

f33 = R3ee'*fee;
n33 = R3ee'*nee + cross(-P3ee, f33);

f22 = R23'*f33;
n22 = R23'*n33 + cross(-P23, f22);

f11 = R12'*f22;
n11 = R12'*n22 + cross(-P12, f11);

f00 = R01'*f11;
n00 = R01'*n11 + cross(-P01, f00)

J = jacobian(n00, fee)

% In the same way, since we propagate backwards here I thought I should 
% have reversed the rotation matrices and the positional vectors

%% Substituting Values

J0 = subs(J, [theta1, theta2, theta3, L1, L2, L3], [0, 0, pi/2, 1, 1, 1])

%% Using jacob functions

clear vars; clear robot
which robot
which jacob0

link1 = link([0 0 0 0], 'modified');
link2 = link([0 0 1 pi/2], 'modified');
link3 = link([pi/2 0 1 0], 'modified');
link4 = link([0 0 1 0], 'modified');
robot = robot({link1 link2 link3 link4});

Q = [0 0 pi/2 0];

J = jacob0(robot, Q);
J = J(1:3, :)

J = jacobn(robot, Q);
J = J(1:3, :)

%% 7. FANUC Configuration
clear robot

alpha   = [0 -pi/2 0 -pi/2 pi/2 -pi/2];         % Link Twist
a       = [0 50 440 35 0 0];        % Link Length
theta   = [0 -pi/2 0 0 0 0];       % Joint Angle
d      = [330 0 0 420 0 80];    % Link Offset

link1 = link([theta(1) d(1) a(1) alpha(1)], 'modified');
link2 = link([theta(2) d(2) a(2) alpha(2)], 'modified');
link3 = link([theta(3) d(3) a(3) alpha(3)], 'modified');
link4 = link([theta(4) d(4) a(4) alpha(4)], 'modified');
link5 = link([theta(5) d(5) a(5) alpha(5)], 'modified');
link6 = link([theta(6) d(6) a(6) alpha(6)], 'modified');

robot2 = robot({link1 link2 link3 link4 link5 link6});
Q = [0 0 0 0 0 0];

J = jacob0(robot2, Q);
J = J(1:3, :)

% Only joint 1 contributes towards the y direction

Q = [0 0 0 0 pi/2 0];

J = jacobn(robot2, Q);
J = J(1:3, :)

% Joint 4 will only affect the y axis and joint 6 will have no effect 
% at the end effector

% Should have used the full 6x6 Jacobian for a 6DOF robot