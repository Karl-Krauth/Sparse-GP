% Exports the results in ICML 2016 paper to be read by R
clear all; clc;
load('all_results_seismic.mat', 'mcmc'); 
cHeader = {'mean_depth_0' 'mean_depth_1' 'mean_depth_2' 'mean_depth_3', ...
            'mean_vel_0',  'mean_vel_1', 'mean_vel_2', 'mean_vel_3' ...
            'std_depth_0' 'std_depth_1' 'std_depth_2' 'std_depth_3', ...
            'std_vel_0',  'std_vel_1', 'std_vel_2', 'std_vel_3' ...
            };
textHeader = strjoin(cHeader, ',');
fid = fopen('mcmc_results.csv','w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);

data = [mcmc.meanH', mcmc.meanV', mcmc.stdH', mcmc.stdV'];

%write data to end of file
dlmwrite('mcmc_results.csv', data, '-append');



