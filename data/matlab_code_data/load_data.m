for iii =1:5
    [x,y,xt,yt] = readHousing(iii);
    csvwrite(['../boston_housing/train_', num2str(iii), '.csv'], [y,x])
    csvwrite(['../boston_housing/test_', num2str(iii), '.csv'], [yt,xt])
end

for iii =1:5
    [x,y,xt,yt] = readBreastCancer(iii);
    csvwrite(['../wisconsin_cancer/train_', num2str(iii), '.csv'], [y,x])
    csvwrite(['../wisconsin_cancer/test_', num2str(iii), '.csv'], [yt,xt])
end

[x,y] = get_mine_data();
csvwrite('../mining/data.csv', [y;x]')

SEED = 1110;
for iii=5:10
    [x,y,xt,yt,model] = getDataAndModel(9);
    csvwrite(['../abalone/train_', num2str(iii), '.csv'], [y,x])
    csvwrite(['../abalone/test_', num2str(iii), '.csv'], [yt,xt])
end

SEED = 1110;
for iii=1:20
    [x,y,xt,yt,model] = getDataAndModel(8);
    csvwrite(['../creep/train_', num2str(iii), '.csv'], [y,x])
    csvwrite(['../creep/test_', num2str(iii), '.csv'], [yt,xt])
end

for iii=1:5 
    [x,y,xt,yt] = readUSPS(iii,[4,7,9]);
    csvwrite(['../USPS/train_', num2str(iii), '.csv'], [y,x])
    csvwrite(['../USPS/test_', num2str(iii), '.csv'], [yt,xt])
end