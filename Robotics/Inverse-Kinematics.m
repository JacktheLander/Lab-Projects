%%% Jack Landers - Robotics

%%% Inverse Kinematics
%% 7.

clearvars; clc;
which robot

% DH Parameters (from 9/26 in class example)
alpha   = [0 0 0];        % Link Twist
a       = [0 1 1];        % Link Length
theta   = [0 0 0];        % Joint Angles
d       = [0 0 0];        % Link Offsets

link1 = link([theta(1) d(1) a(1) alpha(1)], 'modified');
link2 = link([theta(2) d(2) a(2) alpha(2)], 'modified');
link3 = link([theta(3) d(3) a(3) alpha(3)], 'modified');

robot = robot({link1 link2 link3})
mask = [1 1 0 0 0 1];

%% Ikine test - I was previously using a different version of Peter Corke's
%               Robotics toolbox which was great for looping links, but set 
%               me back a long way since the ikine function was broken

T = [1 0 0 2;
    0 1 0 0;
    0 0 1 0;
    0 0 0 1;];

q = [0 0 0];
Q = ikine(robot, T, q, mask);

%% 7. Pos 1, -1
fprintf("\n1, -1\n");

T = [0 1 0 1;
    -1 0 0 -1;
    0 0 1 0;
    0 0 0 1;];

q = [-pi/2 0 0];
Q = ikine(robot, T, q, mask)

q = [0 -pi/2 0];
Q = ikine(robot, T, q, mask)

%% 7. Pos 1.75, 1.5
fprintf("\n1.75, 1.5\n");

distance = sqrt(1.75^2 + 1.5^2)
fprintf("> 2 => Endpoint out of range\n");

%% 7. Pos -0.134, -0.5
fprintf("\n-0.134, -0.5\n");

T = [0 1 0 -0.134;
    -1 0 0 -0.5;
    0 0 1 0;
    0 0 0 1;];

q = [-pi/4 -pi/2 0];
Q = ikine(robot, T, q, mask)

q = [pi 3*pi/4 0];
Q = ikine(robot, T, q, mask)

%% 7. Pos 0, 1
fprintf("\n0, 1\n");

T = [0 1 0 0;
    -1 0 0 1;
    0 0 1 0;
    0 0 0 1;];

q = [pi/4 7*pi/4 0];
Q = ikine(robot, T, q, mask)

q = [3*pi/4 -pi/4 0];
Q = ikine(robot, T, q, mask)

%% 8/9.

clearvars; clc;

% DH Parameters from HW2 Q7 Solutions
% - When I plot Q = 0, I do not get the same result as the answer key

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

robot = robot({link1 link2 link3 link4 link5 link6});
plot(robot, theta)

mask = [1 1 1 1 1 1];

%% Case A
T = [1 0 0 470;
    0 1 0 0;
    0 0 1 725;
    0 0 0 1];

q = theta + [pi/4 pi/4 pi/4 0 -pi/4 0];    % Joint Angles

Q = ikine(robot, T, q, mask)
