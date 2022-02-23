clear all
clc

a = load('d20220215_09o04_genTrajs.mat');
b = load('E:\Research_Data\3DoF_RigidBody\Code_coupled_constmass\TrajectoryGeneration\TestTrajectories\d20220211_13o04_genTrajs.mat');
c = load('xData.mat')

ICsa = reshape(a.stateOut(1,2:7,:),[6,500]);
ICsb = reshape(b.stateOut(1,2:7,:),[6,500]);
xData = c.xData';

aValsWithMatches = zeros(1,500);
bValsWithMatches = zeros(1,500);

for i = 1:500
   aValsWithMatches(i) = any(ICsa(1,i) == xData(1,:));
   bValsWithMatches(i) = any(ICsb(1,i) == xData(1,:)); 
    
end

anyA = any(aValsWithMatches)
anyB = any(bValsWithMatches)