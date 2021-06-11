%% Open Loop application of optimal trajectory generated in OpenOCL
clear all
close all
clc

%%

d = load('ToOrigin_Trajectories/d20210611_12o24_genTrajs.mat');
trajToRun = 1;

% Extract Profiles for Trajectory
ctrlProfile = d.ctrlOut(:,:,trajToRun);
trajFromOCL = d.stateOut(:,2:8,trajToRun);
times = d.stateOut(:,1,trajToRun);

maxStep = diff(times);
maxStep = floor(log10(maxStep(1)));
maxStep = 10^(maxStep-2);

% Set up OpenLoop Simulation
initial_state = trajFromOCL(1,:);
x0 = initial_state;

% Interpolants
method = 'linear'; % This one is bad
% method = 'previous'; % This one is good
u1 = griddedInterpolant(times,ctrlProfile(:,1),method);
u2 = griddedInterpolant(times,ctrlProfile(:,2),method);
u3 = griddedInterpolant(times,ctrlProfile(:,3),method);
% u1 = griddedInterpolant([0,100],[0,0],method);
% u2 = griddedInterpolant([0,100],[0,0],method);
% u3 = griddedInterpolant([0,100],[0,0],method);

options = odeset('MaxStep',maxStep);
[t,x] = ode45(@(t,x) dynamics_openloop(t,x,u1,u2,u3), [times(1),times(end)], x0, options);
xout = x;
tout = t;

u1plot = u1(t);
u2plot = u2(t);
u3plot = u3(t);


%% Plotting

%%%%%%%%%%% POSITIONS %%%%%%%%%%%%%%%
figure;
sgtitle('State')

subplot(3,1,1)
plot(times,trajFromOCL(:,1))
hold on
plot(tout,xout(:,1),'--')
legend('OpenOCL','Open Loop')
ylabel('x_E [m]')

subplot(3,1,2)
plot(times,trajFromOCL(:,2))
hold on
plot(tout,xout(:,2),'--')
legend('OpenOCL','Open Loop')
ylabel('y_E [m]')

subplot(3,1,3)
plot(times,trajFromOCL(:,3))
hold on
plot(tout,xout(:,3),'--')
legend('OpenOCL','Open Loop')
xlabel('Time [s]')
ylabel('\phi [rad]')


%%%%%%%%%%% BODY RATES %%%%%%%%%%%%%%%
figure;
sgtitle('Rates')

subplot(3,1,1)
plot(times,trajFromOCL(:,4))
hold on
plot(tout,xout(:,4),'--')
legend('OpenOCL','Open Loop')
ylabel('\dot{x} [m/s]')

subplot(3,1,2)
plot(times,trajFromOCL(:,5))
hold on
plot(tout,xout(:,5),'--')
legend('OpenOCL','Open Loop')
ylabel('\dot{y} [m/s]')

subplot(3,1,3)
plot(times,trajFromOCL(:,6))
hold on
plot(tout,xout(:,6),'--')
legend('OpenOCL','Open Loop')
xlabel('Time [s]')
ylabel('\dot{\phi} [rad/s]')


%%%%%%%% CONTROLS %%%%%%%%%%%%%%%
figure;
sgtitle('Controls')

subplot(3,1,1)
plot(times,ctrlProfile(:,1))
hold on;
plot(t,u1plot,'--')
legend('OpenOCL','Open Loop')
ylabel('F_x [N]')

subplot(3,1,2)
plot(times,ctrlProfile(:,2))
hold on;
plot(t,u2plot,'--')
legend('OpenOCL','Open Loop')
ylabel('F_y [N]')

subplot(3,1,3)
plot(times,ctrlProfile(:,3))
hold on;
plot(t,u3plot,'--')
legend('OpenOCL','Open Loop')
ylabel('M [N-m]')





