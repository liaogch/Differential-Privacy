
opt_epsilon_max_check_lambda = [0.0049 0.0042525 0.00461]
shape_gamma_check_lambda  = [19.5174 4.8793 1.6264]
scale_gamma_check_lambda  = [0.1 0.4 1.2]

epsilon_ref = 0;
W = 0.03;
c=1;
m = 100;
alpha = 0.5;
beta = par_gam_beta(1)*par_gam_beta(2);

%%

gd1 = makedist('gamma','a',shape_gamma_check_lambda(1),'b',scale_gamma_check_lambda(1));
x = 0:0.01:6;
y = pdf(gd1,x);
figure;
plot(x,y,'k','LineWidth',2);
xlabel('\lambda','FontSize',15);
ylabel('pdf(\lambda)','FontSize',15);
title('Var(\lambda)=0.1952','FontSize',15)

epsilon_max = opt_epsilon_max_check_lambda(1);
lambda_search = 0.1:0.01:6;
for j = 1:length(lambda_search)
    if check_participation( lambda_search(j),beta,alpha,epsilon_ref,W,c,epsilon_max,m)==0
        lambda_target(1) = lambda_search(j);
        break;
    end
end
set(gca,'Xtick',[0  1.0  2.0  lambda_target(1)  4.0 5.0 6.0])
lambda_plot = 0:0.01:lambda_target(1);
hold on;
fill([lambda_plot,fliplr(lambda_plot)],[pdf(gd1,lambda_plot),16*zeros(1,length(lambda_plot))],'c');


%%

gd2 = makedist('gamma','a',shape_gamma_check_lambda(2),'b',scale_gamma_check_lambda(2));
x = 0:0.01:6;
y = pdf(gd2,x);
figure;
plot(x,y,'k','LineWidth',2);
xlabel('\lambda','FontSize',15);
ylabel('pdf(\lambda)','FontSize',15);
title('Var(\lambda)=0.7807','FontSize',15)
epsilon_max = opt_epsilon_max_check_lambda(2);
lambda_search = 0.1:0.01:6;

for j = 1:length(lambda_search)
    if check_participation( lambda_search(j),beta,alpha,epsilon_ref,W,c,epsilon_max,m)==0
        lambda_target(2) = lambda_search(j);
        break;
    end
end
set(gca,'Xtick',[0  1.0  2.0  lambda_target(2)  4.0 5.0 6.0])
lambda_plot = 0:0.01:lambda_target(2);
hold on;
fill([lambda_plot,fliplr(lambda_plot)],[pdf(gd2,lambda_plot),16*zeros(1,length(lambda_plot))],'c');

%%

gd3 = makedist('gamma','a',shape_gamma_check_lambda(3),'b',scale_gamma_check_lambda(3));
x = 0:0.01:6;
y = pdf(gd3,x);
figure;
plot(x,y,'k','LineWidth',2);
xlabel('\lambda','FontSize',15);
ylabel('pdf(\lambda)','FontSize',15);
title('Var(\lambda)=2.3421','FontSize',15);
epsilon_max = opt_epsilon_max_check_lambda(3);
lambda_search = 0.1:0.01:6;

for j = 1:length(lambda_search)
    if check_participation( lambda_search(j),beta,alpha,epsilon_ref,W,c,epsilon_max,m)==0
        lambda_target(3) = lambda_search(j);
        break;
    end
end
set(gca,'Xtick',[0  1.0  2.0  lambda_target(3)  4.0 5.0 6.0 7.0 8.0 9.0])
lambda_plot = 0:0.01:lambda_target(3);
hold on;
fill([lambda_plot,fliplr(lambda_plot)],[pdf(gd3,lambda_plot),16*zeros(1,length(lambda_plot))],'c');


%%
opt_epsilon_max_check_beta = [0.0012,0.0010,0.001111666666667]
shape_gamma_check_beta  = [7502.12765957447,468.882978723404,133.966565349544]
scale_gamma_check_beta  = [0.0001,0.0016,0.0056]


epsilon_ref = 0;
W = 0.02;
c=2.5;
m = 100;
alpha = 0.5;
%beta = par_gam_beta(1)*par_gam_beta(2);
lambda = par_gam_lambda(1)*par_gam_lambda(2);
beta_target= zeros(1,length(opt_epsilon_max_check_beta));

%%

gd1 = makedist('gamma','a',shape_gamma_check_beta(1),'b',scale_gamma_check_beta(1));
x = 0.6:0.001:0.9;
y = pdf(gd1,x);
figure;
plot(x,y,'k','LineWidth',2);
axis([0.6 0.9 0 50]);
xlabel('\beta','FontSize',15);
ylabel('pdf(\beta)','FontSize',15);
title('Var(\beta)=7.5\times 10^{-5}','FontSize',15)

epsilon_max = opt_epsilon_max_check_beta(1);
beta_search = 0.9:-0.001:0.6;
for j = 1:length(beta_search)
    if check_participation( lambda,beta_search(j),alpha,epsilon_ref,W,c,epsilon_max,m)==0
        beta_target(1) = beta_search(j);
        break;
    end
end
set(gca,'Xtick',[0.6 beta_target(1)  0.80 0.90])
beta_plot = beta_target(1):0.001:0.8;
hold on;
fill([beta_plot,fliplr(beta_plot)],[pdf(gd1,beta_plot),zeros(1,length(beta_plot))],'c');



%%

gd2 = makedist('gamma','a',shape_gamma_check_beta(2),'b',scale_gamma_check_beta(2));
x = 0.6:0.001:0.9;
y = pdf(gd2,x);
figure;
plot(x,y,'k','LineWidth',2);
axis([0.6 0.9 0 12]);
xlabel('\beta','FontSize',15);
ylabel('pdf(\beta)','FontSize',15);
title('Var(\beta)=1.2\times 10^{-3}','FontSize',15)
epsilon_max = opt_epsilon_max_check_beta(2);
beta_search = 0.9:-0.001:0.6;

for j = 1:length(beta_search)
    if check_participation( lambda,beta_search(j),alpha,epsilon_ref,W,c,epsilon_max,m)==0
        beta_target(2) = beta_search(j);
        break;
    end
end
set(gca,'Xtick',[0.60 0.65 beta_target(2) 0.75 0.80 0.85 0.90])
beta_plot = beta_target(2):0.001:0.9;
hold on;
fill([beta_plot,fliplr(beta_plot)],[pdf(gd2,beta_plot),16*zeros(1,length(beta_plot))],'c');


%%

gd3 = makedist('gamma','a',shape_gamma_check_beta(3),'b',scale_gamma_check_beta(3));
x = 0.6:0.001:0.9;
y = pdf(gd3,x);
figure;
plot(x,y,'k','LineWidth',2);
axis([0.6 0.9 0 7]);
xlabel('\beta','FontSize',15);
ylabel('pdf(\beta)','FontSize',15);

title('Var(\beta)=4.2\times 10^{-3}','FontSize',15);
epsilon_max = opt_epsilon_max_check_beta(3);
beta_search = 0.9:-0.001:0.6;

for j = 1:length(beta_search)
    if check_participation( lambda,beta_search(j),alpha,epsilon_ref,W,c,epsilon_max,m)==0
        beta_target(3) = beta_search(j);
        break;
    end
end
set(gca,'Xtick',[0.60 0.65 beta_target(3) 0.80 0.85 0.90 ])
beta_plot = beta_target(3):0.001:0.9;
hold on;
fill([beta_plot,fliplr(beta_plot)],[pdf(gd3,beta_plot),16*zeros(1,length(beta_plot))],'c');
