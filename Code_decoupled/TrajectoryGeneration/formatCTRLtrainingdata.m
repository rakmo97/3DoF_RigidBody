%% Prepares data for ANN training
clear all
close all
clc


%% Setup Directory
directory = 'ToOrigin_Trajectories';
addpath(directory);
datadir = dir(directory);
filenames = {datadir.name};
datafiles = filenames(3:end);

directory = 'ToSurface_Trajectories';
addpath(directory);
datadir = dir(directory);
filenames = {datadir.name};
datafiles2 = filenames(3:end);

datafiles =  {[datafiles,datafiles2]};
datafiles = datafiles{1};


%% Pull data and format

% Preallocation loop
disp('Preallocating');
numdata = 0;
for i = 1:length(datafiles)
    d = load(datafiles{i});
    
    lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
    if isempty(lastidx)
        lastidx = size(d.Jout,1);
    end
    
    numinthisfile = size(d.ctrlOut,1)*lastidx;
    numdata = numdata + numinthisfile;  
    
end


% Preallocate
Xfull_2 = zeros(numdata,7);
tfull_2 = zeros(numdata,3);
times = zeros(numdata,1);

count = 1;
for i = 1:length(datafiles)

    d = load(datafiles{i});
    
    disp(['Extracting datafile ',num2str(i),' of ',num2str(length(datafiles))]);
    
    lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
    if isempty(lastidx)
        lastidx = size(d.Jout,1);
    end
    
    
    for j = 1:lastidx

        for k = 1:100
        
            Xfull_2(count,:) = [d.stateFinal(j,1)-d.stateOut(k,2,j),... % x
                d.stateFinal(j,2)-d.stateOut(k,3,j),... % y
                d.stateFinal(j,3)-d.stateOut(k,4,j),... % phi
                d.stateFinal(j,4)-d.stateOut(k,5,j),... % dx
                d.stateFinal(j,5)-d.stateOut(k,6,j),... % dy
                d.stateFinal(j,6)-d.stateOut(k,7,j),... % dphi
                d.stateOut(k,8,j),...
                ];
            
            tfull_2(count,:) = [d.ctrlOut(k,1,j),d.ctrlOut(k,2,j),d.ctrlOut(k,3,j)];
            
            times(count) = d.stateOut(k,1,j);
            
            if(rem(count,100)==0)
                tfull_2(count,:) = [0,0,0];
%                 disp(count)
            end
            
            count = count+1;
        end

    end
end
disp('Done extracting')
disp(['Full dataset size: ',num2str(count-1)])


%% Separate into training and testing data
disp('Separating training and testing data')

num2train = 300000;
Xtrain2 = Xfull_2(1:num2train,:);
ttrain2 = tfull_2(1:num2train,:);
Xtest2 = Xfull_2(num2train+1:end,:);
ttest2 = tfull_2(num2train+1:end,:);
times_train = times(1:num2train,:);
times_test = times(num2train+1:end,:);
disp('Done separating')

%% Save data to .mat file
disp('Saving data to mat file')

save('../NetworkTraining/ANN2_data.mat','Xfull_2','tfull_2','Xtrain2','ttrain2','Xtest2','ttest2','times_train','times_test','times');

disp('Saved')