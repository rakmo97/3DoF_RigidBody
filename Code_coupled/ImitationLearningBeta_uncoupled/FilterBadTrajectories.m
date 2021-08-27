%%
clear all
close all
clc

%%

% Get User Input
userinput = input("Script will DELETE files. Continue? [Y/N]: ",'s');

if strcmp(userinput,"Y")
    disp("Continuing with script to delete files")
else
    error("EXITING SCRIPT")
end


% Load in Files
directory = 'QueriedTrajectories';
addpath(directory);
datadir = dir(directory);
filenames = {datadir.name};
datafiles = filenames(3:end);


% Look through files to find which to delete
idx_to_delete = [];
count = 1;
for i = 1:length(datafiles)

    d = load(datafiles{i});

    disp(['Checking datafile ',num2str(i),' of ',num2str(length(datafiles))]);

    lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
    if isempty(lastidx)
        lastidx = size(d.Jout,1);
    end


    for j = 1:lastidx
        if any(idx_to_delete == i)
            break
        end

        for k = 1:100
            
            if d.stateOut(k,3,j) < 0
                idx_to_delete = [idx_to_delete, i];
                disp(['--Adding ', datafiles{i},' to trash list'])
                break
            end

            count = count+1;
        end
    end
end


filestodelete = {datafiles{idx_to_delete}};




% Get User Input
disp("Script will DELETE files NOW (no turning back).")
userinput = input("To Continue type destroy: ",'s');

if strcmp(userinput,"destroy")
    disp("Continuing with script to delete files")
else
    error("EXITING SCRIPT")
end


for i = 1:length(filestodelete)
    warning(['--Deleting ', datafiles{i}])
    delete([directory,'/',filestodelete{i}])
end
disp("Done")



