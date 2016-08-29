function [x,y,xt,yt] = readHousing(i)
% Generates a training/testing split for this dataset.
if nargin == 0
  i = 1;
end
seeds = [1110 1234 9022 4355 2341 6328 9876 32323 4966 3663];
rng(seeds(i),'twister');
data = load('housing.data');
Ntrain = 300;
randind = randperm(size(data,1));
x = data(randind(1:Ntrain),1:13);
y = data(randind(1:Ntrain),end);
xt = data(randind(Ntrain+1:end),1:13);
yt = data(randind(Ntrain+1:end),end);

[x,xmean,xstd] = standardize(x,[],[]);
xt = standardize(xt,xmean,xstd);
end

