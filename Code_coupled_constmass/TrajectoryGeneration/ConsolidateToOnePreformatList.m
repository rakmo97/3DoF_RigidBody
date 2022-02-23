%% Consolidates Data to one pre-formatted list
clear all
close all
clc


%% Setup Directory
directory = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/ToOrigin_Trajectories';
addpath(directory);
datadir = dir(directory);
filenames = {datadir.name};
datafiles = filenames(3:end);

directory = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/ToSurface_Trajectories';
addpath(directory);
datadir = dir(directory);
filenames = {datadir.name};
datafiles2 = filenames(3:end);

datafiles =  {[datafiles,datafiles2]};
datafiles = datafiles{1};

%% Pull data and consolidate
Jout = [];
conf = [];
ctrlOut = [];
objectiveOut = [];
runTimeOut = [];
stateFinal = [];
stateOut = [];
surfFunctionOut = [];

for i = 1:length(datafiles)

    d = load(datafiles{i});
    lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
    if isempty(lastidx)
        lastidx = size(d.Jout,1);
    end
    
    Jout = cat(1,Jout,d.Jout(1:lastidx,:));
    conf = cat(1,conf,d.conf);
    ctrlOut = cat(3,ctrlOut,d.ctrlOut(:,:,1:lastidx));
    objectiveOut = cat(1,objectiveOut,d.objectiveOut(1:lastidx,:));
    stateFinal = cat(1,stateFinal,d.stateFinal(1:lastidx,:));
    stateOut = cat(3,stateOut,d.stateOut(:,:,1:lastidx));
    surfFunctionOut = cat(1,surfFunctionOut,d.surfFunctionOut(1:lastidx,:));

end

%% Save file
saveout = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/TrajectoryGeneration/consolidatedPreformat.mat';
disp(['Filename: ',saveout])
save(saveout,'surfFunctionOut','objectiveOut','Jout','stateOut','ctrlOut','runTimeOut','stateFinal','conf');
fprintf("\nProgram Complete!\n")




