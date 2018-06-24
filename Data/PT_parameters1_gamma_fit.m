a = load('D:\liao\MATLAB\Differential Privacy\Data\PT_Parameters1.txt');

alpha = a(:,1);
alpha_adj = alpha(alpha<2);

beta = a(:,2);
beta_adj = beta(beta<2);

lambda = a(:,3);
lambda_adj = lambda(lambda<6);

par_gam_alpha = gamfit(alpha_adj);
par_gam_beta = gamfit(beta_adj);
par_gam_lambda = gamfit(lambda_adj);

[h_alpha,p_alpha] = chi2gof(alpha_adj,'cdf',{@gamcdf,par_gam_alpha(1),par_gam_alpha(2)},'Alpha',0.01);
[h_beta,p_beta] = chi2gof(beta_adj,'cdf',{@gamcdf,par_gam_beta(1),par_gam_beta(2)},'Alpha',0.01);
[h_lambda,p_lambda] = chi2gof(lambda_adj,'cdf',{@gamcdf,par_gam_lambda(1),par_gam_lambda(2)},'Alpha',0.01);














