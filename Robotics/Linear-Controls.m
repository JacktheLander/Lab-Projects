%%% Jack Landers - Robotics II

%%% Linear Controls

%% 3. Plotting Position of Mass Spring Damper
t = linspace(0, 5, 1000);

x = (7/3)*exp(-2*t) - (4/3)*exp(-5*t);

figure;
plot(t, x);
grid on;

xlabel('t (s)');
ylabel('x(t)');
title('Position of Mass Spring Damper');
