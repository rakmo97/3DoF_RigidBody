%% Open Loop application of optimal trajectory generated in OpenOCL
clear all
close all
clc

%%
trajToRun = 1;
% Extract Profiles for Trajectory

d = load('../NetworkTraining/ANN2_data.mat');
idxs = ((trajToRun-1)*100+1):(((trajToRun-1)*100+1)+99);
ctrlProfile = d.tfull_2(idxs,:);
trajFromOCL= d.Xfull_2(idxs,:);
trajFromOCL(:,1:6) = -trajFromOCL(:,1:6);
times = d.times(idxs);


% d = load('ToOrigin_Trajectories/d20210512_15o18_genTrajs.mat');
% % d = load('ToOrigin_Trajectories/d20210512_15o21_genTrajs.mat');
% ctrlProfile = d.ctrlOut(:,:,trajToRun);
% trajFromOCL = d.stateOut(:,2:8,trajToRun);
% times = d.stateOut(:,1,trajToRun);

maxStep = diff(times);
maxStep = floor(log10(maxStep(1)));
maxStep = 10^(maxStep-2);

% Set up OpenLoop Simulation
x0 = trajFromOCL(1,:);
xout = zeros(100,7);
xout(1,:) = x0;
tout = zeros(100,1);


for i = 1:length(times)-1

    Fx = ctrlProfile(i,1);
    Fy = ctrlProfile(i,2);
    options = odeset('MaxStep',maxStep);
    [t,x] = ode45(@(t,x) dynamics_openloop(t,x,Fx,Fy), [times(i),times(i+1)], xout(i,:), options);
    xout(i+1,:) = x(end,:);
    tout(i+1) = t(end);
    
end



%% Plotting

%%%%%%%%%%% POSITIONS %%%%%%%%%%%%%%%
figure;
sgtitle('State')

subplot(3,1,1)
plot(times,trajFromOCL(:,1))
hold on
plot(tout,xout(:,1),'--')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('x_E [m]')

subplot(3,1,2)
plot(times,trajFromOCL(:,2))
hold on
plot(tout,xout(:,2),'--')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('y_E [m]')

subplot(3,1,3)
plot(times,trajFromOCL(:,3))
hold on
plot(tout,xout(:,3),'--')
legend('OpenOCL','Open Loop1','Open Loop2')
xlabel('Time [s]')
ylabel('\phi [rad]')


%%%%%%%%%%% BODY RATES %%%%%%%%%%%%%%%
figure;
sgtitle('Rates')

subplot(3,1,1)
plot(times,trajFromOCL(:,4))
hold on
plot(tout,xout(:,4),'--')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('\dot{x} [m/s]')

subplot(3,1,2)
plot(times,trajFromOCL(:,5))
hold on
plot(tout,xout(:,5),'--')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('\dot{y} [m/s]')

subplot(3,1,3)
plot(times,trajFromOCL(:,6))
hold on
plot(tout,xout(:,6),'--')
legend('OpenOCL','Open Loop1','Open Loop2')
xlabel('Time [s]')
ylabel('\dot{\phi} [rad/s]')


%%%%%%%% CONTROLS %%%%%%%%%%%%%%%
figure;
sgtitle('Controls')

subplot(2,1,1)
plot(times,ctrlProfile(:,1))
hold on;
plot(times,ctrlProfile(:,1),'--')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('F_x [N]')

subplot(2,1,2)
plot(times,ctrlProfile(:,2))
hold on;
plot(times,ctrlProfile(:,2),'--')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('F_y [N]')






