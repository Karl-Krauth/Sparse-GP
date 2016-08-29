function [x,y,xt,yt] = readBreastCancer(i)
% Generates a training/testing split for this dataset.
if nargin == 0
  i = 1;
end
seeds = [1110 1234 9022 4355 2341 6328 9876 32323 4966 3663];
rng(seeds(i),'twister');
zz = csvread('breast_cancer_wisconsin.data');
Y =  zz(:, 11);
X = zz(:, 2:10);
Y(Y==2) = 1; 
Y(Y==4) = -1;
Ntrain = 300;
randind = randperm(size(X,1));
x = X(randind(1:Ntrain),:);
y = Y(randind(1:Ntrain));
xt = X(randind(Ntrain+1:end),:);
yt = Y(randind(Ntrain+1:end));

[x,xmean,xstd] = standardize(x,[],[]);
xt = standardize(xt,xmean,xstd);

end

