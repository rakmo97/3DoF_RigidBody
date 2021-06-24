%% Prepares data for ANN training
clear all
close all
clc


%% Setup Directory
directory = 'ToSurface_Trajectories';
addpath(directory);
datadir = dir(directory);
filenames = {datadir.name};
datafiles = filenames(3:end);

directory = 'obsolete_Trajectories/ToSurface_Trajectories';
addpath(directory);
datadir = dir(directory);
filenames = {datadir.name};
datafiles2 = filenames(3:end);

datafiles =  {[datafiles,datafiles2]};
datafiles = datafiles{1};


% Preallocation loop
disp('Preallocating');
numdata = 0;
for i = 1:length(datafiles)
    d = load(datafiles{i});
    
    lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
    if isempty(lastidx)
        lastidx = size(d.Jout,1);
    end
    
    numinthisfile = lastidx;
    numdata = numdata + numinthisfile;  
    
end


n_grid = 200;
n_obj = 2;
n_init = 7;

n_inputA = n_obj + n_init;
n_inputB = n_grid;
n_output = 2;

% Preallocate
Xfull_1A = zeros(numdata,n_inputA);
Xfull_1B = zeros(numdata,n_inputB,2);
tfull_1 = zeros(numdata,n_output);
times = zeros(numdata,1);



gridX = linspace(-100,100,n_grid);
gridY = zeros(numdata,n_grid);
surfXfull = zeros(numdata,3*n_grid);
surfYfull = zeros(numdata,3*n_grid);

count = 1;

for i = 1:length(datafiles)
    d = load(datafiles{i});
    
    lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
    if isempty(lastidx)
        lastidx = size(d.Jout,1);
    end
    
    for j = 1:lastidx
        surfXfull(count,:) = linspace(-100,100,3*n_grid);
        surfYfull(count,:) = d.surfFunctionOut{j}(surfXfull(count,:));
        gridYnonoise = d.surfFunctionOut{j}(gridX);
        gridY(count,:) = gridYnonoise + 2*randn(size(gridYnonoise));
        
        obj = d.objectiveOut(j,:);
        init = d.stateOut(1,2:end,j);
        Xfull_1A(count,:) = [obj,init];
        Xfull_1B(count,:,:) = reshape([gridX;gridY(count,:)]',[1,n_inputB,2]);
        
        tfull_1(count,:) = d.stateOut(end,2:3,j);

        count = count + 1;

    end
    

end

disp(['Full dataset size: ',num2str(count-1)])



%% Separate into training and testing data
disp('Separating training and testing data')

num2train = 4200;
Xtrain_1A = Xfull_1A(1:num2train,:);
Xtrain_1B = Xfull_1B(1:num2train,:,:);
ttrain1 = tfull_1(1:num2train,:);
Xtest_1A = Xfull_1A(num2train+1:end,:);
Xtest_1B = Xfull_1B(num2train+1:end,:,:);
ttest1 = tfull_1(num2train+1:end,:);
surfXtrain = surfXfull(1:num2train,:);
surfYtrain = surfYfull(1:num2train,:);
surfXtest = surfXfull(num2train+1:end,:);
surfYtest = surfYfull(num2train+1:end,:);

disp('Done separating')

%% Save data to .mat file
disp('Saving data to mat file')

save('../NetworkTraining/ANN1_data.mat','Xfull_1A','Xfull_1B','tfull_1',...
    'Xtrain_1A','Xtrain_1B','ttrain1','Xtest_1A','Xtest_1B','ttest1',...
    'surfXfull','surfYfull','surfXtrain','surfYtrain',...
    'surfXtest','surfYtest');

disp('Saved')

