%%% Jack Landers - Robotics II

%%% Trajectory Generation

%% Comparing Methods

% RR robot - trajectory generation example
hold off
axis
clear all
RoneLink;
% Establish a time vector
t=[0:.25:5];
% Define initial and final robot joint space poses
Qi=[-pi/3]; %These are the values for the lecture example
Qf=[pi/3];

% Generate a cubic trajectory plan
a0 = -60; a1 = 0; a2 = 14.4; a3 = -1.92;
q = a0 + a1*t + a2*t.^2 + a3*t.^3;
qd = a1 + 2*a2*t + 3*a3*t.^2;
qdd = 2*a2 + 6*a3*t;
% Show how angles, velocities, accelerations evolve
subplot(3,1,1), plot(t, q)
subplot(3,1,2), plot(t, qd)
subplot(3,1,3), plot(t, qdd)

%% Generate a trajectory plan with jtraj
[q qd qdd]=jtraj(Qi, Qf, t);
% Show how angles, velocities, accelerations evolve
subplot(3,1,1), plot(t, q)
subplot(3,1,2), plot(t, qd)
subplot(3,1,3), plot(t, qdd)

%% Trajectory Generation Example

%RR robot - trajectory generation example
hold off
axis
clear all
SCURRtwolink;
plotoption=5; 
r = RRtwolink
%establish a time vector
t=[0:.5:10];

%define initial and final robot joint space poses
Qi=[3*pi/8;-pi/2]; %These are the values for the lecture example
Qf=[5*pi/8;pi/2];

%generate a trajectory plan
[q qd qdd]=jtraj(Qi, Qf, t);

%show how angles, velocities, accele  rations evolve
if plotoption==1
subplot(3,1,1), plot(t, q)
subplot(3,1,2), plot(t, qd)
subplot(3,1,3), plot(t, qdd)
end

%show how robot moves in a 'straight line' in joint space
if plotoption==2
plot(q*[1;0],q*[0;1])
end

%show how the motion is not 'straight' in operational space
TJ=fkine(r, q)
Ree=transl(TJ);
if plotoption==3
subplot(2,1,1), plot(t,Ree(:,1))
subplot(2,1,2), plot(t,Ree(:,2))
end
if plotoption==4
plot(r, q, 'loop')
end
if plotoption==5
axis('square'); axis([-2 2 -2 2]); axis manual; hold on;
for z=1:1:length(t)
plot(cos(q(z,1)), sin(q(z,1)), 'o')
plot(Ree(z,1), Ree(z,2), 'o')
plot([0;cos(q(z,1));Ree(z,1)],[0;sin(q(z,1));Ree(z,2)])
end
end

%Now let's try doing the planning in Cartesian Space
Tinit=fkine(r,Qi);
Tfinal=fkine(r, Qf);
rr=jtraj(0,1,t);
TC = ctraj(Tinit, Tfinal, rr);

%Endpoint x and y coordinates over time
if plotoption==6
plot(t, transl(TC)); grid;
end

%Endpoint in Cartesian space
k=transl(TC);
if plotoption==7
axis('square'); axis([-2 2 -2 2]); grid
plot(k(:,1),k(:,2))
end

%The resulting joint angles
Q=ikine(r, TC, [0;pi/2], [1 1 0 0 0 0])

%Which from an overhead view gives
if plotoption==8  
axis('square'); axis([-2 2 -2 2]); axis manual; hold on;
for z=1:1:length(t)
plot(cos(Q(z,1)), sin(Q(z,1)), 'o')
plot(k(z,1), k(z,2), 'o')
plot([0;cos(Q(z,1));k(z,1)],[0;sin(Q(z,1));k(z,2)])
pause
end
end