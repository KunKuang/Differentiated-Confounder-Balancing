function [ W, ATE, ATE_residual ] = DCB_with_setting_parameters( X_t, X_c, Y_t, Y_c )
%Differentied confounder balancing 

lambda0 = 10;
lambda1 = 10;
lambda2 = 100;
lambda3 = 1;
lambda4 = 0.1;
MAXITER = 1000;
ABSTOL = 1e-6;

[ ATE, W, beta ] = DCB(X_t, X_c, Y_t, Y_c, lambda0, lambda1, lambda2, lambda3, lambda4, MAXITER, ABSTOL );
W = W.*W;

ATE_residual = mean(Y_t) - (mean(X_t,1)*beta+W'*(Y_c-X_c*beta));

end

