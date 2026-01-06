%%% Jack Landers - Robotics

%%% Homework 2
%% 1.

clearvars; clc;
syms q1 q2 q3 q4 l1 l2 l3

% DH Parameters 
alpha   = [0 0 0 0];        % Link Twist
a       = [0 10 5 10];      % Link Length
theta   = [pi/2 -pi/2 pi/2 0];    % Joint Angles
d       = [0 0 0 0];        % Link Offsets

Links(1:4) = Link();

for i = 1:4
    Links(i) = Link([theta(i) d(i) a(i) alpha(i)], 'modified');
end

robot = SerialLink(Links, 'name', '1.');

T = simplify(robot.fkine([q1 q2 q3 q4]))

Q = [pi/2 -pi/2 pi/2 0]; 
robot.fkine(Q)
plot(robot, Q)

%% 2.

clearvars; clc;
syms q1 q2 q3 l1 l2 l3 l4

% DH Parameters 
alpha2   = [0 pi/2 0 0];         % Link Twist
a2       = [0 0 l3 l4];        % Link Length
theta2   = [q1 q2 q3 0];       % Joint Angle
d2       = [l1+l2 0 0 0];    % Link Offset

Links(1:4) = Link();

for i = 1:4
    Links(i) = Link([theta2(i) d2(i) a2(i) alpha2(i)], 'modified');

    T = simplify(Links(i).A(theta2(i)));

    fprintf("T%d = \n", i);
    T
end

robot2 = SerialLink(Links, 'name', '2.');
fprintf("To-ee = \n");
Toee = simplify(robot2.fkine(theta2))

clearvars; clc;

l1 = 1; l2 = 1; l3 = 2; l4 = 1;
q1 = pi/2; q2 = pi/2; q3 = pi/2;

% DH Parameters 
alpha2   = [0 pi/2 0 0];         % Link Twist
a2       = [0 0 l3 l4];        % Link Length
theta2   = [q1 q2 q3 0];       % Joint Angle
d2       = [l1+l2 0 0 0];    % Link Offset

Links(1:4) = Link();

for i = 1:4
    Links(i) = Link([theta2(i) d2(i) a2(i) alpha2(i)], 'modified');

    T = simplify(Links(i).A(theta2(i)));

    fprintf("T%d = \n", i);
    T
end

robot2 = SerialLink(Links, 'name', '2.');
fprintf("To-ee = \n");
Toee = simplify(robot2.fkine(theta2))
vpa(Toee, 6)
theta2   = [0 0 0 0];
plot(robot2, theta2)

%% 3.

clearvars; clc;
syms q1 q3 l1 l3 d2

% DH Parameters 
alpha   = [0 pi/2 q3 pi];         % Link Twist
a       = [0 l1 l3 0];        % Link Length
theta   = [q1 0 pi/2 0];       % Joint Angle
d      = [0 d2 0 0];    % Link Offset

Links(1:4) = Link();

for i = 1:4
    Links(i) = Link([theta(i) d(i) a(i) alpha(i)], 'modified');

    T = simplify(Links(i).A(theta(i)));

    fprintf("T%d = \n", i);
    T
end

robot = SerialLink(Links, 'name', '3.');
Toee = simplify(robot.fkine(theta));
fprintf("To-ee = \n");
Toee = simplify(Toee)


% Calculating for [90, 2, 90], L1=2, L3=1
clearvars; clc;

q1 = pi/2; q3 = pi/2;
l1 = 2; l3 =1;
d2 = 2;

% DH Parameters 
alpha   = [0 pi/2 q3 pi];         % Link Twist
a       = [0 l1 l3 0];        % Link Length
theta   = [q1 0 pi/2 0];       % Joint Angle
d      = [0 d2 0 0];    % Link Offset

Links(1:4) = Link();

for i = 1:4
    Links(i) = Link([theta(i) d(i) a(i) alpha(i)], 'modified');

    T = simplify(Links(i).A(theta(i)));

    fprintf("T%d = \n", i);
    T
end

robot = SerialLink(Links, 'name', '3.');
Toee = simplify(robot.fkine(theta));
fprintf("To-ee = \n");
Toee = simplify(Toee)
plot(robot, theta)


%% 5.

clearvars; clc;
syms alpha a theta d

Ts = [1 0 0 0; 
    0 cos(alpha) -sin(alpha) 0; 
    0 sin(alpha) cos(alpha) 0; 
    0 0 0 1]

Tq = [1 0 0 a; 
    0 1 0 0; 
    0 0 1 0;
    0 0 0 1]

Tr = [cos(theta)   -sin(theta)  0  0;
      sin(theta)   cos(theta)  0  0;
      0            0           1  0;
      0            0           0  1]

Ti = [1 0 0 0; 
    0 1 0 0; 
    0 0 1 d; 
    0 0 0 1]

Tee = Ts * Tq * Tr * Ti

link = Link([theta d a alpha], 'modified');
T = simplify(link.A(theta))

%% 7. 

clearvars; clc;

% DH Parameters 
alpha   = [0 -pi/2 0 -pi/2 pi/2 -pi/2];         % Link Twist
a       = [0 50 440 35 0 0];        % Link Length
theta   = [0 pi/2 0 0 0 0];       % Joint Angle
d      = [330 0 0 420 0 80];    % Link Offset

link1 = link([theta(1) d(1) a(1) alpha(1)], 'modified');
link2 = link([theta(2) d(2) a(2) alpha(2)], 'modified');
link3 = link([theta(3) d(3) a(3) alpha(3)], 'modified');
link4 = link([theta(4) d(4) a(4) alpha(4)], 'modified');
link5 = link([theta(5) d(5) a(5) alpha(5)], 'modified');
link6 = link([theta(6) d(6) a(6) alpha(6)], 'modified');

robot = robot({link1 link2 link3 link4 link5 link6});
plot(robot, theta)

%Q = theta + [pi/2 -pi/4 0 -pi/2 0 0];
%plot(robot, Q)