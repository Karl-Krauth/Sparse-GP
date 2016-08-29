load sarcos_inv
load sarcos_inv_test
x = sarcos_inv(:,1:21);
y = sarcos_inv(:,22:end);

xtest = sarcos_inv_test(:,1:21);
ytest = sarcos_inv_test(:,22:end);

csvwrite(['../sarcos/train_all', '.csv'], [y,x])
csvwrite(['../sarcos/test_all', '.csv'], [ytest,xtest])


outputs = [4,7];
y = y(:,outputs);
ytest = ytest(:,outputs);
[x,xmean,xstd] = standardize(x,[],[]);
xtest = standardize(xtest,xmean,xstd);
csvwrite(['../sarcos/train_', '.csv'], [y,x])
csvwrite(['../sarcos/test_', '.csv'], [ytest,xtest])
