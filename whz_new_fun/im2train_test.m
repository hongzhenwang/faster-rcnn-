clc;clear;
%%将图片分为train和test，train：test=3:1
imfold='/home/whz/image_set/Plan_extract20151011/image';
trainfold='/home/whz/image_set/Plan_extract20151011/image_train';
testfold='/home/whz/image_set/Plan_extract20151011/image_test';
imdir=dir(fullfile(imfold,'*.jpg'));
txtfilepath='/home/whz/image_set/Plan_extract20151011/';
TrainTxt=fopen([txtfilepath,'plane_trainval.txt'],'wt');
TestTxt=fopen([txtfilepath,'plane_test.txt'],'wt');
for i=1:length(imdir)
    if mod(i,3)==0
%         I=imread(fullfile(imfold,imdir(i).name));
%         imwrite(I,fullfile(testfold,imdir(i).name));
        imname=imdir(i).name;
        fprintf(TestTxt,'%s\n',imname(1:end-4));
    else
%         I=imread(fullfile(imfold,imdir(i).name));
%         imwrite(I,fullfile(trainfold,imdir(i).name));
        imname=imdir(i).name;
        fprintf(TrainTxt,'%s\n',imname(1:end-4));
    end
end
fclose(TrainTxt);
fclose(TestTxt);   

