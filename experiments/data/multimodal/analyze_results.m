function analyze_results()

X = load('X.txt');
Y = load('Y.txt');
Xtest = load('Xtest.txt');
Ytest = f(Xtest);
pred_mean = load('pred_mean.txt');
F = load('post_samples.txt'); % S x N
F = F';

% Samples from posterior
%h = plot(Xtest, F);
%for i = 1 : size(F,2)
%    h(i).Color = [0, 0, 1, 0.2]; % last value in vector is transparency
%end
%hold on;



plot(X, Y, 'bo');
hold on;
plot(Xtest, Ytest, 'k.');
plot(Xtest, pred_mean, 'rx')


end

function val = h(x)
val = exp(- x.^2) .* (2.0 * x);
end

function val = f(x)
val = h(h(x));
end

