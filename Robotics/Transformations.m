%%% Jack Landers - Robotics

%%% Homework 1
%% 2.

a = [1, 2, 3];
b = [1; 2; 3;];
C = [1, 1, 1; 1, 2, 2; 1, 2, 3;];
D = [2, 1, 3; 2, 1, 1; 2, 2, 2;];

fprintf("2.\na)");
a*b

fprintf("b)");
b*a

fprintf("c)");
a*C 

fprintf("d) Dimensions don't match\n");
% C*a

fprintf("e) Dimensions don't match\n");
% b*D

fprintf("f)");
D*b

fprintf("g)");
C*D

fprintf("h)");
D*C

%% 3.

R = [-1, 0, 0; 0, 0, 1; 0, 1, 0;];
Paz = [1, 1, 1]';
Pbo = [1; 0; 1;];

fprintf("3.");
P = R*Paz + Pbo

%% 4.

Rx = rotx(90)
Rz = rotz(90)

R12 = Rx*Rz

Pa = [1, 1, 0]';
Pb = R12*Pa

%% 7.

Tab = [-1, 0, 0, 3; 0, -1, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1;];
Tbc = [0, sin(30), -cos(30), 0; 0, -cos(30), -sin(30), 0; -1, 0, 0, 2; 0, 0, 0, 1;];

Tab*Tbc

%% 9.

a = 53.13;
Tab = [-1, 0, 0, 0; 0, 0, -1, 4; 0, -1, 0, 2; 0, 0, 0, 1;];
Tbc = [-cos(a), -cos(a), 0, 3; sin(a), 0, 0, 0; 0, sin(a), 0, 0; 0, 0, 0, 1];

Tab*Tbc
