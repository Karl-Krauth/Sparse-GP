function [x,y,xt,yt] = readUSPS(i,digits)
%READUSPS [x,y,xt,yt] = readUSPS(i,digitss)
%   Reads USPS data.
%
seeds = [1110 1234 9022 4355 2341 6328 9876 32323 4966 3663];
rng(seeds(i),'twister');
load usps_resampled
x = [train_patterns'; test_patterns'];
y = [train_labels'; test_labels'];
y(y == -1) = 0;
labels = labelsFromBinary(y)-1;
ally = []; % all y that are in digits
allx = [];
for j=1:numel(digits)
  instances = sum(labels == digits(j));
  ally = [ally; oneOfK(j*ones(instances,1),numel(digits))]; 
  allx = [allx; x(labels == digits(j),:)];
end
allx = standardize(allx,[],[]);
% shuffle
ind = randperm(size(allx,1));
allx = allx(ind,:);
ally = ally(ind,:);
% split into training / testing
ntrain = ceil(0.5*size(allx,1));
x = allx(1:ntrain,:);
y = ally(1:ntrain,:);
xt = allx(ntrain+1:end,:);
yt = ally(ntrain+1:end,:);
end

