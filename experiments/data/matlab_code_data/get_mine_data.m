function [xx, yy] = get_mine_data(bin_width)
%GET_MINE_DATA 
%
%     [xx, yy] = get_mine_data(bin_width)
%
% Inputs:
%     bin_width 1x1 Optional, Default=50.
%                   Number of days in each bin (except possibly the last)
%
% The default bin width of 50 gives 811 bins.
%
% Outputs:
%            xx 1xN Centres of bins. Time measured in days.
%                   (Could argue about best definition here. I've picked it so
%                   that if a bin only contains first day the bin is at '1', if
%                   it contains the first two days it is at '1.5' and so on.)
%            yy 1xN Number of events in each bin

% Iain Murray, October 2009

if ~exist('bin_width', 'var')
    bin_width = 50;
end

% Facts from paper (could be derived from data, but I'm using to sanity check):
num_days = 40550;
num_events = 191;
intervals = load('mining.dat');
event_days = [1, cumsum(intervals(:)')+1];
assert(event_days(end) == num_days);

edges = [1:bin_width:num_days, num_days+1];
bin_counts = histc(event_days, edges);
assert(sum(bin_counts) == num_events);
% Should have no data at exactly num_days+1, also strip off this cruft:
assert(bin_counts(end) == 0);
bin_counts = bin_counts(1:end-1);

xx = (edges(1:end-1) + (edges(2:end)-1)) / 2;
yy = bin_counts;
