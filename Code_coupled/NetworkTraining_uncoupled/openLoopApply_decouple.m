%% Open Loop application of optimal trajectory generated in OpenOCL
clear all
close all
clc

%%

trajToRun = 1;
d = load("ANN2_data.mat");
idxs = ((trajToRun-1)*100+1):((trajToRun-1)*100+100);
ctrlProfile = d.tfull_2(idxs,:);
ctrlProfileOrig = d.t_orig(idxs,:);
trajFromOCL = d.Xfull_2(idxs,:);
times = d.times(idxs);


% Extract Profiles for Trajectory
d = load('../TrajectoryGeneration/ToOrigin_Trajectories/d20210604_11o05_genTrajs.mat');
% ctrlProfile = d.ctrlOut(:,:,trajToRun);
% trajFromOCL = d.stateOut(:,2:8,trajToRun);
% times = d.stateOut(:,1,trajToRun);
target = d.stateFinal(trajToRun,:);


% Simulation Setup/Parameters
x0 = trajFromOCL(1,:);

n_times = length(times);
nState = 7;
nCtrlc = 2;
nCtrld = 3;

t = zeros(size(times));
phiSave = zeros(size(times));
condNumbers = zeros(size(times));
x = zeros(n_times,nState);
Fi = zeros (n_times, nCtrld);
TxTyMsave = zeros(n_times,nCtrld);
FxFysave = zeros(n_times,nCtrlc);
r = 1;



%% Run Simualation
x(1,:) = x0;

for i = 1:n_times-1
    
    phi = x(i,3);
    D = [cos(phi),-sin(phi);sin(phi),cos(phi);r,0];
    condNumbers(i) = cond(D);

    TxTyMsave(i,:) = D*ctrlProfileOrig(i,:)';

    Fi(i,:) = (TxTyMsave(i,:))';
%     Fi(i,:) = (ctrlProfile(i,:))';

    
    u1 = Fi(i,1); u2 = Fi(i,2); u3 = Fi(i,3);
    [tsol, xsol] = ode45(@(t,x) dynamics_decoupled(t,x,u1,u2,u3), [times(i),times(i+1)], x(i,:));
    x(i+1,:) = xsol(end,:);
        
    
    
end

%% Evaluate

J_ANN = calculatePathCost(times, Fi)
J_OCL = calculatePathCost(times, ctrlProfileOrig)


%% Plotting

%%%%%%%%%%% POSITIONS %%%%%%%%%%%%%%%

figure;
sgtitle('Landing Profile')

plot(trajFromOCL(:,1), trajFromOCL(:,2))
plot(x(:,1),x(:,2),'--')
xlabel('x [m]')
ylabel('y [m]')
legend('OpenOCL','Open Loop')


figure;
sgtitle('State')

subplot(3,1,1)
plot(times,trajFromOCL(:,1))
hold on
plot(times,x(:,1),'--')
legend('OpenOCL','Open Loop')
ylabel('x_E [m]')

subplot(3,1,2)
plot(times,trajFromOCL(:,2))
hold on
plot(times,x(:,2),'--')
legend('OpenOCL','Open Loop')
ylabel('y_E [m]')

subplot(3,1,3)
plot(times,trajFromOCL(:,3))
hold on
plot(times,x(:,3),'--')
legend('OpenOCL','Open Loop')
xlabel('Time [s]')
ylabel('\phi [rad]')


%%%%%%%%%%% BODY RATES %%%%%%%%%%%%%%%
figure;
sgtitle('Rates')

subplot(3,1,1)
plot(times,trajFromOCL(:,4))
hold on
plot(times,x(:,4),'--')
legend('OpenOCL','Open Loop')
ylabel('\dot{x} [m/s]')

subplot(3,1,2)
plot(times,trajFromOCL(:,5))
hold on
plot(times,x(:,5),'--')
legend('OpenOCL','Open Loop')
ylabel('\dot{y} [m/s]')

subplot(3,1,3)
plot(times,trajFromOCL(:,6))
hold on
plot(times,x(:,6),'--')
legend('OpenOCL','Open Loop')
xlabel('Time [s]')
ylabel('\dot{\phi} [rad/s]')


%%%%%%%% MASS %%%%%%%%%%%%%%%
figure;
sgtitle('Mass')

plot(times, trajFromOCL(:,7))
hold on
plot(times, x(:,7),'--')
xlabel('Time [s]')
ylabel('Mass [kg]')
legend('OpenOCL','Open Loop')



%%%%%%%% CONTROLS Coupled %%%%%%%%%%%%%%%
figure;
sgtitle('Controls Coupled')

subplot(2,1,1)
plot(times,ctrlProfileOrig(:,1))
hold on;
plot(times,Fi(:,1),'--')
legend('OpenOCL','Open Loop')
ylabel('F_x [N]')

subplot(2,1,2)
plot(times,ctrlProfileOrig(:,2))
hold on;
plot(times,Fi(:,2),'--')
legend('OpenOCL','Open Loop')
ylabel('F_y [N]')


%%%%%%%% CONTROLS Uncoupled %%%%%%%%%%%%%%%

figure;
sgtitle('Controls Uncoupled')

subplot(3,1,1)
plot(times,ctrlProfile(:,1))
hold on;
plot(times,TxTyMsave(:,1),'--')
legend('OpenOCL','Open Loop')
ylabel('F_x [N]')

subplot(3,1,2)
plot(times,ctrlProfile(:,2))
hold on;
plot(times,TxTyMsave(:,2),'--')
legend('OpenOCL','Open Loop')
ylabel('F_y [N]')

subplot(3,1,3)
plot(times,ctrlProfile(:,3))
hold on;
plot(times,TxTyMsave(:,3),'--')
legend('OpenOCL','Open Loop')
ylabel('F_y [N]')



%% Functions

function Jout = calculatePathCost(t,u)
    
L = zeros(size(t));

for i = 1:size(u,1)
    L(i) = 0;
    for j = 1:size(u,2)
        L(i) = L(i) + u(i,j)^2;
    end
end

Jout = trapz(t,L);


end


