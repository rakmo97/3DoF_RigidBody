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
numtrajs = numdata/100;

num_final_masses = 1000;
final_masses = linspace(0,500,num_final_masses);

% Preallocate
% Xfull_2 = zeros(numdata+numtrajs,13);
% tfull_2 = zeros(numdata+numtrajs,6);
% times = zeros(numdata+numtrajs,1);
% Xfull_2 = zeros(numdata+num_final_masses,13);
% tfull_2 = zeros(numdata+num_final_masses,6);
% times = zeros(numdata+num_final_masses,1);
Xfull_2 = zeros(numdata,13);
tfull_2 = zeros(numdata,6);
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
        
            
            if(rem(count,101)==0)
                state_bound = 1e-1;
                Xfull_2(count,:) = [(-state_bound + (state_bound - -state_bound) * rand(1,12)),d.stateOut(100,14,j)];
                tfull_2(count,:) = [0,0,0,0,0,0];
                times(count) = d.stateOut(100,1,j) + (d.stateOut(100,1,j)-d.stateOut(99,1,j));
            else
                
                Xfull_2(count,:) = [d.stateFinal(j,1)-d.stateOut(k,2,j),... % x
                    d.stateFinal(j,2)-d.stateOut(k,3,j),... % y
                    d.stateFinal(j,3)-d.stateOut(k,4,j),... % z
                    d.stateFinal(j,4)-d.stateOut(k,5,j),... % dx
                    d.stateFinal(j,5)-d.stateOut(k,6,j),... % dy
                    d.stateFinal(j,6)-d.stateOut(k,7,j),... % dz
                    d.stateFinal(j,7)-d.stateOut(k,8,j),... % phi
                    d.stateFinal(j,8)-d.stateOut(k,9,j),... % theta
                    d.stateFinal(j,9)-d.stateOut(k,10,j),... % psi
                    d.stateFinal(j,10)-d.stateOut(k,11,j),... % p
                    d.stateFinal(j,11)-d.stateOut(k,12,j),... % q
                    d.stateFinal(j,12)-d.stateOut(k,13,j),... % r
                    d.stateOut(k,14,j),...
                    ];

                tfull_2(count,:) = [d.ctrlOut(k,1,j),d.ctrlOut(k,2,j),d.ctrlOut(k,3,j),...
                                    d.ctrlOut(k,4,j),d.ctrlOut(k,5,j),d.ctrlOut(k,6,j)];

                times(count) = d.stateOut(k,1,j);
            end
            
            
            count = count+1;
        end

    end
end
% 
% for i = 1:num_final_masses
%    
%     state_bound = 1e-1;
%     Xfull_2(count,:) = [(-state_bound + (state_bound - -state_bound) * rand(1,12)),final_masses(i)];
%     tfull_2(count,:) = [0,0,0,0,0,0];
% %     times(count) = d.stateOut(100,1,j) + (d.stateOut(100,1,j)-d.stateOut(99,1,j));
%     times(count) = 80;
%     count = count + 1;
% 
% end

disp('Done extracting')
disp(['Full dataset size: ',num2str(count-1)])


%% Separate into training and testing data
disp('Separating training and testing data')

num2train = 1450000;
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