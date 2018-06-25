function [ ATT, W, beta ] = DCB(X_t, X_c, Y_t, Y_c, lambda0, lambda1, lambda2, lambda3, lambda4, MAXITER, ABSTOL )
% Copyright by Kun Kuang
% Estimate the ATE from observational data via Differentiated Confounder Balancing (DCB) algorithm
% The optimized objective function as follow:
%J = lambda0*(beta.*(mean_X_t-X_c'*(W.*W)))'*(beta.*(mean_X_t-X_c'*(W.*W)))+lambda1*sum((Y_c-X_c*beta).^2)+lambda2*((W.*W)'*(W.*W))+lambda3*sum(beta.^2)+lambda4*abs(beta);
%
%%INPUT
%X_t (n_t*p): observationalvariables of treated units
%X_c (n_c*p): observationalvariables of control units
%Y_t (n_t*1): Observed outcome of treated units
%Y_c (n_c*1): Observed outcome of control units
%lambda0, lambda1, lambda2, lambda3, lambda4: hyper-parameters
%MAX_ITER:
%ABSTOL: 
%
%%OUTPUT
%ATT: Average Treatment effect on Treated
%W (n*1), optimized sample weight for units in control group
%beta (p*1), optimized confounder weight for differentiating confounders

m = size(X_t,1);
n = size(X_c,1);
if size(X_t,2) == size(X_c,2)
    p = size(X_t,2);
else
    fprintf('error: Dimensions of X_t and X_c are not equal!');
end
mean_X_t = mean(X_t,1)';

%% paramters initialization
W = ones(n,1)./n;
W_prev = W;
beta = ones(p,1);
beta_prev = beta;

parameter_iter = 0.5;
J_loss = ones(MAXITER, 1)*(-1);

lambda_W = 1;
lambda_beta = 1;

%% proximal gradient algorithm
for iter = 1:MAXITER
    %beta
    y = beta;
    beta = beta+(iter/(iter+3))*(beta-beta_prev);
    f_base = f_function(W, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3);
    while 1
        grad_beta = 2*lambda0*(beta'*(mean_X_t-X_c'*(W.*W))*(mean_X_t-X_c'*(W.*W)))-2*lambda1*X_c'*((1+W.*W).*(Y_c-X_c*beta))+2*lambda3*beta;
        z = prox_l1(beta-lambda_beta*grad_beta, lambda_beta*lambda4);
        if f_function(W, z, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3) <= f_base + grad_beta'*(z-beta) + (1/(2*lambda_beta))*sum((z-beta).^2)
            break;
        end
        lambda_beta = parameter_iter*lambda_beta;
    end
    beta_prev = y;
    beta = z;
    
    %W
    y = W;
    W = W+(iter/(iter+3))*(W-W_prev);
    f_base = f_function(W, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3);
    while 1
        grad_W = -4*lambda0*(beta'*(mean_X_t-X_c'*(W.*W)))*X_c*beta.*W+2*lambda1*((Y_c-X_c*beta).^2).*W+4*lambda2*W.*W.*W;
        z = prox_l1(W-lambda_W*grad_W, 0);
        if f_function(z, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3) <= f_base + grad_W'*(z-W) + (1/(2*lambda_W))*sum((z-W).^2)
            break;
        end
        lambda_W = parameter_iter*lambda_W;
    end
    W_prev = y;
    W = z;
    W = W./sqrt(sum(W.*W));
	
	ATT = mean(Y_t) - (W.*W)'*Y_c;
    
    J_loss(iter) = f_function(W, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3) + lambda4*sum(abs(beta));
    if iter > 1 && abs(J_loss(iter) - J_loss(iter-1)) < ABSTOL
        fprintf('Get the optimal results at iteration %d\n', iter);
        fprintf('our iteration %d ... J_error: %f, lambda0: %f, lambda1: %f, lambda3: %f, lambda4: %f\n', iter, J_loss(iter), lambda0*(beta.*(mean_X_t-X_c'*(W.*W)))'*(beta.*(mean_X_t-X_c'*(W.*W))), lambda1*sum((Y_c-X_c*beta).^2), lambda2*((W.*W)'*(W.*W))+lambda3*sum(beta.^2), lambda4*sum(abs(beta)));
        break
    elseif iter == MAXITER
        fprintf('our iteration %d ... J_error: %f, lambda0: %f, lambda1: %f, lambda3: %f, lambda4: %f\n', iter, J_loss(iter), lambda0*(beta.*(mean_X_t-X_c'*(W.*W)))'*(beta.*(mean_X_t-X_c'*(W.*W))), lambda1*sum((Y_c-X_c*beta).^2), lambda2*((W.*W)'*(W.*W))+lambda3*sum(beta.^2), lambda4*sum(abs(beta)));
    end
    
end

    function f_x = f_function(W, beta, mean_X_t, X_c, Y_c, lambda0, lambda1, lambda2, lambda3)
        f_x = lambda0*(beta'*(mean_X_t-X_c'*(W.*W)))'*(beta'*(mean_X_t-X_c'*(W.*W)))+lambda1*(1+W.*W)'*((Y_c-X_c*beta).^2)+lambda2*((W.*W)'*(W.*W))+lambda3*sum(beta.^2);

    end

end
