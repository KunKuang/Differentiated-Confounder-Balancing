clear;
warning off all;
%clc;

fw = fopen('results.txt','wb');

for e = [3];
for m_n = [2000,5000];
for p = [50,100];
for rate = [0.2,0.8]; % confounding rate
for t_rate = [1.0] %logitc rate

ATE_list = [];
ATT_error_list = [];
for experiment_iter = 1:100

    fprintf('\n n = %d, p = %d, e = %d, rate = %.2f, t_rate = %.2f, Experimental iteration number: %d ......\n',m_n, p, e, rate, t_rate, experiment_iter);
    
    X = normrnd(0,1,m_n,p);
    
    % set T with linear function (misspecified function)
    p_t = int64(rate*p);
    T = X(:,1:p_t)*ones(p_t,1)*t_rate+normrnd(0,1,m_n,1);
    T(T<0) = 0;
    T(T>0) = 1;
    
    % set Y with linear function
    CE = 1;
    epsilon = normrnd(0,e,m_n,1);
    Y = CE*T+epsilon;
    Y1 = CE+epsilon;
    Y0 = epsilon;
    for iter = 1:p
        weight = 0;
          if mod(iter,2) ==0
            weight = iter/2;
          end
        Y = Y+weight*X(:,iter)+X(:,iter).*T;
        Y1 = Y1+weight*X(:,iter)+X(:,iter);
        Y0 = Y0+weight*X(:,iter);
    end
    
    % ATT ground truth
    ATT_gt = mean(Y1(T==1)-Y0(T==1));

    X_t = X(T==1,:); X_c = X(T==0,:);
    Y_t = Y(T==1,:); Y_c = Y(T==0,:);

    m = size(X_t,1);
    n = size(X_c,1);
    
    % direct estimator
    ATE_naive = mean(Y_t)-mean(Y_c);
    
    % Our Differentiated Confounder Balancing estimator
    [ W_dcb, ATE_dcb, ATE_dcb_regression ] = DCB_with_setting_parameters( X_t, X_c, Y_t, Y_c );
    
    ATT_error = [ATE_naive, ATE_dcb, ATE_dcb_regression] - ATT_gt;
    ATT_error_list = [ATT_error_list;ATT_error];
  
end

error_mean = mean(ATT_error_list,1);
error_std = std(ATT_error_list,1,1);
error_mae = mean(abs(ATT_error_list),1);
error_rmse = sqrt(mean((ATT_error_list).^2,1));

fprintf('\n m_n: %d ... p: %d ... e=%d ... rate=%.2f ... t_rate=%.2f \n', m_n, p, e, rate, t_rate);
fprintf('ATE_naive, ATE_dcb, ATE_dcb_regression\n');
fprintf('Bias: %s\n', num2str(error_mean,'%.4f\t'));
fprintf('SD  : %s\n', num2str(error_std,'%.4f\t'));
fprintf('MAE : %s\n', num2str(error_mae,'%.4f\t'));
fprintf('RMSE: %s\n', num2str(error_rmse,'%.4f\t'));

fprintf(fw,'\n m_n: %d ... p: %d ... e=%d ... rate=%.2f ... t_rate=%.2f \n', m_n, p, e, rate, t_rate);
fprintf(fw,'ATE_naive, ATE_dcb, ATE_dcb_regression\n');
fprintf(fw,'Bias: %s\n', num2str(error_mean,'%.4f\t'));
fprintf(fw,'SD  : %s\n', num2str(error_std,'%.4f\t'));
fprintf(fw,'MAE : %s\n', num2str(error_mae,'%.4f\t'));
fprintf(fw,'RMSE: %s\n', num2str(error_rmse,'%.4f\t'));

end
end
end
end
end

fclose(fw);