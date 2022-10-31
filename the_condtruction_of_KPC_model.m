clear all;clc;
% The construction framework of KPC model
%% input variable
% dim          The dimension of the input variable
% funaqq       The function used to compute the true response
% x_train      The training set of the model
% y_train      The real response corresponding to the training set
% x_predict    The test set of the model
% y_predict    The real response corresponding to the test set
[N dim]=size(x_train)
funaqq=@fun_F1;
%% 1 Latin Hypercube obtains the training and test sets
dim=10;Num=10*dim;
%   Input variable dimension
%   The number of sample points is 10 times the number of dimensions(10*dim)
x_train=lhsdesign(Num,dim);
for i=1:Num
   Y_train(i,1)=funaqq(x_train(i,:));
end
%   The number of sample points in the test set is 20*dim
x_test=lhsdesign(Num,dim);
for i=1:Num
   Y_test(i,1)=funaqq(x_test(i,:));
end
%% 2 Calculating projected correlation/Distance Correlation/MIC
% The computation time is closely related to the number of samples. When the number of training sets is large, a small number of samples are used to replace them.
% The construction of KPC needs to ensure that x_train has no duplicate data
    if N>50
       xpc=x_train(1:50,:);ypc=y_train(1:50,:);
       R_kpc=PC_XG(xpc,ypc);
    else
       R_kpc=PC_XG(x_train,y_train);
    end
%% 2 The construction of KPC model
[KPC_G,~] = modelfit_KPC(x_train, y_train, @regpoly0, @corrgauss, R_kpc);
%% 3 The prediction based on KPC model
[Y_predict MSE] = predictor(x_predict, KPC_G);



%% 1 Calculating projected correlation
% The computation time is closely related to the number of samples. When the number of training sets is large, a small number of samples are used to replace them.
% The construction of KPC needs to ensure that x_train has no duplicate data
    if N>50
       xpc=x_train(1:50,:);ypc=y_train(1:50,:);
       R_kpc=PC_XG(xpc,ypc);
    else
       R_kpc=PC_XG(x_train,y_train);
    end
%% 2 The construction of KPC/KDIC/KMIC/DACE model
[KPC_G,~] = modelfit_KPC(x_train, y_train, @regpoly0, @corrgauss, R_kpc);
% The first call to the function takes extra time, so a repetitive modeling process is added
tic;
[KPC_G,~] = modelfit_KPC(x_train, y_train, @regpoly0, @corrgauss, R_kpc);
time_kpc=toc;tic
[KPC_G,~] = modelfit_KDIC(x_train, y_train, @regpoly0, @corrgauss, lob, upb, R_dic);
time_kdic=toc;tic
[KPC_G,~] = modelfit_KPC(x_train, y_train, @regpoly0, @corrgauss, R_kpc);
time_kmic=toc;tic
[KPC_G,~] = modelfit_KPC(x_train, y_train, @regpoly0, @corrgauss, R_kpc);
time_dace=toc;
%% 3 The prediction based on KPC model
[Y_predict MSE] = predictor(x_predict, KPC_G);