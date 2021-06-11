%% Open Loop application of optimal trajectory generated in OpenOCL
clear all
close all
clc

%%
trajToRun = 1;
% Extract Profiles for Trajectory

d = load('../NetworkTraining/ANN2_data.mat');
idxs = ((trajToRun-1)*100+1):(((trajToRun-1)*100+1)+99);
ctrlProfile1 = d.tfull_2(idxs,:);
trajFromOCL1 = d.Xfull_2(idxs,:);
trajFromOCL1(:,1:6) = -trajFromOCL1(:,1:6);
times1 = d.times(idxs);


d = load('ToOrigin_Trajectories/d20210604_11o05_genTrajs.mat');
ctrlProfile2 = d.ctrlOut(:,:,trajToRun);
trajFromOCL2 = d.stateOut(:,2:8,trajToRun);
times2 = d.stateOut(:,1,trajToRun);

maxStep = diff(times1);
maxStep = floor(log10(maxStep(1)));
maxStep = 10^(maxStep-2);

% Set up OpenLoop Simulation
x01 = trajFromOCL1(1,:);
xout1 = zeros(100,7);
xout1(1,:) = x01;
tout1 = zeros(100,1);

x02 = trajFromOCL2(1,:);
xout2 = zeros(100,7);
xout2(1,:) = x02;
tout2 = zeros(100,1);

for i = 1:length(times1)-1

    Fx1 = ctrlProfile1(i,1);
    Fy1 = ctrlProfile1(i,2);
    options = odeset('MaxStep',maxStep);
    [t,x] = ode45(@(t,x) dynamics_openloop(t,x,Fx1,Fy1), [times1(i),times1(i+1)], xout1(i,:), options);
    xout1(i+1,:) = x(end,:);
    tout1(i+1) = t(end);
%     a = x(end,:)
    
    Fx2 = ctrlProfile2(i,1);
    Fy2 = ctrlProfile2(i,2);
    options = odeset('MaxStep',maxStep);
    [t,x] = ode45(@(t,x) dynamics_openloop(t,x,Fx2,Fy2), [times2(i),times2(i+1)], xout2(i,:), options);
    xout2(i+1,:) = x(end,:);
    tout2(i+1) = t(end);
%     b = x(end,:)
%     c = a - b
end



%% Plotting

%%%%%%%%%%% POSITIONS %%%%%%%%%%%%%%%
figure;
sgtitle('State')

subplot(3,1,1)
plot(times1,trajFromOCL1(:,1))
hold on
plot(tout1,xout1(:,1),'--')
plot(tout2,xout2(:,1),'-x')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('x_E [m]')

subplot(3,1,2)
plot(times1,trajFromOCL1(:,2))
hold on
plot(tout1,xout1(:,2),'--')
plot(tout2,xout2(:,2),'-x')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('y_E [m]')

subplot(3,1,3)
plot(times1,trajFromOCL1(:,3))
hold on
plot(tout1,xout1(:,3),'--')
plot(tout2,xout2(:,3),'-x')
legend('OpenOCL','Open Loop1','Open Loop2')
xlabel('Time [s]')
ylabel('\phi [rad]')


%%%%%%%%%%% BODY RATES %%%%%%%%%%%%%%%
figure;
sgtitle('Rates')

subplot(3,1,1)
plot(times1,trajFromOCL1(:,4))
hold on
plot(tout1,xout1(:,4),'--')
plot(tout2,xout2(:,4),'-x')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('\dot{x} [m/s]')

subplot(3,1,2)
plot(times1,trajFromOCL1(:,5))
hold on
plot(tout1,xout1(:,5),'--')
plot(tout2,xout2(:,5),'-x')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('\dot{y} [m/s]')

subplot(3,1,3)
plot(times1,trajFromOCL1(:,6))
hold on
plot(tout1,xout1(:,6),'--')
plot(tout2,xout2(:,6),'-x')
legend('OpenOCL','Open Loop1','Open Loop2')
xlabel('Time [s]')
ylabel('\dot{\phi} [rad/s]')


%%%%%%%% CONTROLS %%%%%%%%%%%%%%%
figure;
sgtitle('Controls')

subplot(2,1,1)
plot(times1,ctrlProfile1(:,1))
hold on;
plot(times1,ctrlProfile1(:,1),'--')
plot(times1,ctrlProfile2(:,1),'-x')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('F_x [N]')

subplot(2,1,2)
plot(times1,ctrlProfile1(:,2))
hold on;
plot(times1,ctrlProfile1(:,2),'--')
plot(times1,ctrlProfile2(:,2),'-x')
legend('OpenOCL','Open Loop1','Open Loop2')
ylabel('F_y [N]')






