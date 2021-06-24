
clc; clear;
addpath ..\..\OptimTraj\



p.g0 = 9.81;
p.g = 9.81/6;
p.Isp = 300;
p.r = 1;

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                        Problem Bounds                                   %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% PosVel0 = [2000,2000,0,15,15,0]';
PosVel0 = 1.0e+03*[1.1,1.3,0,0,0,0]';
m0 = 500;
mF = 100; 
Tmax = 400*4*9.81;

P.bounds.initialTime.low = 0;
P.bounds.initialTime.upp = 0;

P.bounds.finalTime.low = 0;
P.bounds.finalTime.upp = 60*60;

P.bounds.state.low = [-5000;-5000;0;-300;-300;-300;mF];
P.bounds.state.upp = [5000;5000;5000;300;300;300;m0];

P.bounds.initialState.low = [PosVel0;m0];
P.bounds.initialState.upp = [PosVel0;m0];

P.bounds.finalState.low = [-0*ones(6,1);mF];
P.bounds.finalState.upp = [0*ones(6,1);m0];

P.bounds.control.low = [-Tmax;0;-Tmax];
P.bounds.control.upp = Tmax*ones(3,1);

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                           Initial Guess                                 %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
P.guess.time = [0, 55];  %(s)
P.guess.state = [ [PosVel0;m0],  [0;0;0;0;0;0;mF] ];
P.guess.control = [Tmax*ones(3,1),[-Tmax;0;-Tmax]];

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                 Objective and Dynamic functions                         %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

% Dynamics function:
P.func.dynamics = @(t,x,u)( dynamics3DOF(x,u,p) );

% Path cost:
P.func.pathObj = @(t,x,u) ((vecnorm(u)).^2);

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                  Options and Method selection                           %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
P.options(1).method = 'trapezoid';
P.options(1).defaultAccuracy = 'low';


P.options(2).method = 'trapezoid';
P.options(2).defaultAccuracy = 'medium';
P.options(2).nlpOpt.MaxFunEvals = 2e5;
P.options(2).nlpOpt.MaxIter = 1e4;
P.options(2).trapezoid.nGrid = 100;

P.options(3).method = 'trapezoid';
P.options(3).defaultAccuracy = 'medium';
P.options(3).nlpOpt.MaxFunEvals = 2e5;
P.options(3).nlpOpt.MaxIter = 1e4;
P.options(3).trapezoid.nGrid = 100;

%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                              Solve!                                     %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
soln = optimTraj(P);

% t = linspace(soln(end).grid.time(1),soln(end).grid.time(end),250);
% x = soln(end).interp.state(t);
% u = soln(end).interp.control(t);

t = soln(end).grid.time;
x = soln(end).grid.state;
u = soln(end).grid.control;


%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                            Plotting                                     %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
close all

figure;
plot(x(1,:),x(2,:))
xlabel('x [m]')
ylabel('y [m]')


figure;
sgtitle('States')
subplot(3,2,1)
plot(t,x(1,:))
ylabel('X [m]')

subplot(3,2,2)
plot(t,x(4,:))
ylabel('X-dot [m/s]')

subplot(3,2,3)
plot(t,x(2,:))
ylabel('Y [m]')

subplot(3,2,4)
plot(t,x(5,:))
ylabel('Y-dot [m/s]')

subplot(3,2,5)
plot(t,x(3,:))
ylabel('\Phi [rad]')
xlabel('Time [s]')

subplot(3,2,6)
plot(t,x(6,:))
ylabel('\Phi-dot [rad/s]')
xlabel('Time [s]')


figure;
plot(t,x(7,:))
xlabel('Time [s]')
ylabel('Mass [kg]')




figure;
sgtitle('Control Strategy')
xlabel('Time [s]')
subplot(3,1,1);
plot(t,u(1,:))
ylabel('T_x [N]')
subplot(3,1,2);
plot(t,u(2,:))
ylabel('T_y [N]')
subplot(3,1,3);
plot(t,u(3,:))
ylabel('T_z [N]')
xlabel('Time [s]')







