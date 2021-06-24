clear all
close all
clc


%%
pathToOptimTraj = "../OptimTraj/";
addpath(pathToOptimTraj);


%%
x0 = [1000;1000;0;0;500];
target = [0;0;0;0];
uMax = 15000;


%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                      Set up function handles                            %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
problem.func.dynamics = @(t,x,u)( dynamics_2DoF(t,x,u) );
problem.func.pathObj = @(t,x,u)( objective(t,x,u) );



%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                 Set up bounds on state and control                      %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

problem.bounds.initialTime.low = 0;
problem.bounds.initialTime.upp = 0;
problem.bounds.finalTime.low = 1;
problem.bounds.finalTime.upp = 150;

problem.bounds.state.low = [-1500;0;-50;-50;0];
problem.bounds.state.upp = [1500;1500;50;50;500];

problem.bounds.initialState.low = [x0];
problem.bounds.initialState.upp = [x0];

problem.bounds.finalState.low = [target;0];
problem.bounds.finalState.upp = [target;500];

problem.bounds.control.low = [-uMax;0];
problem.bounds.control.upp = [uMax;uMax];



%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                 Initialize trajectory with guess                        %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
% Car travels at a speed of one, and drives in a straight line from start
% to finish point.

problem.guess.time = [0, 100];   % time = distance/speed
problem.guess.state = [ [x0], [target;0]];
problem.guess.control = [[0;0],[0;0]];  


%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                      Options for Transcription                          %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

problem.options.nlpOpt = optimset(...
    'display','iter',...
    'MaxFunEval',1e5,...
    'tolFun',1e-6);

problem.options.method = 'trapezoid';
problem.options.trapezoid.nGrid = 100;



%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%                            Solve!                                       %
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

soln = optimTraj(problem);




