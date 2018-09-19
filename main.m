%{
lambda = 1:0.25:4.5;
Opt_U=zeros(1,length(lambda));
Opt_epsilon_max = zeros(1,length(lambda));
for i=1:length(lambda)
    [ Opt_U(i),Opt_epsilon_max(i),~] = ProspectTheoryBased( lambda(i),0.88,0.88,10000,50,0.5,0.5,0,0.0001:0.0001:0.01 );
end

[ ~,Opt_epsilon_EUT_max,~] = ProspectTheoryBased( 1,1,1,10000,50,0.5,0.5,0,0.0001:0.0001:0.01 );

Uc_byEUT=zeros(1,length(lambda));
for i=1:length(lambda)
    [ Uc_byEUT(i),~,~] = ProspectTheoryBased( lambda(i),0.88,0.88,10000,50,0.5,0.5,0,Opt_epsilon_EUT_max);
end

plot(lambda,Opt_U,'*',lambda,Uc_byEUT,'o');
dif=Opt_U-Uc_byEUT;
per = dif./Opt_U*100
%}

%{
x=-0.25:0.001:0.25;
len = length(x);
y1 = zeros(1,len);
y2 = zeros(1,len);
y3 = zeros(1,len);
y4 = zeros(1,len);

for i=1:len
    y1(i)=Valuation_Fun( x(i),1,1,1,0);
end


hold on;
for i=1:len
    y2(i)=Valuation_Fun( x(i),1,1.5,1,0);
end

for i=1:len
    y3(i)=Valuation_Fun( x(i),0.8,1.5,0.8,0);
end

for i=1:len
    y4(i)=Valuation_Fun( x(i),0.5,2,0.5,0);
end

plot(x,y1,'k-',x,y2,'g:',x,y3,'-.',x,y4,'--','LineWidth',2)
legend('\lambda=1,\beta=1','\lambda=1.5,\beta=1','\lambda=1.5,\beta=0.8','\lambda=2,\beta=0.5')
set(gca,'XTick',0)
%}

%{
n_target1 = 2000; %data amount utility is 2/3 at this number
n_target2 = 4000;
n_target3 = 4000;

k1 = 0.989;
k2 = 0.989;
k3 = 0.509;
%k = 0.273;
%g = 0.271;
l1 = 9.8e-4;
l2 = 1.3e-4;
l3 = 1.3e-4;
%l1=(3*k1-1)/n_target1;
%l2=(3*k2-1)/n_target2;
%l3=(3*k3-1)/n_target3;
n = 1:6000;
len = length(n);
R1 = zeros(length(n),1);
R2 = zeros(length(n),1);
R3 = zeros(length(n),1);
for i = 1:len
    R1(i) = Rn(k1,l1,n(i));
    R2(i) = Rn(k2,l2,n(i));
    R3(i) = Rn(k3,l3,n(i));
end
plot(n,R1,'r-',n,R2,'k-.',n,R3,'b--','LineWidth',2)
xlabel('n');
ylabel('R(n)');
set(gca,'FontSize',15);
%}
%{
k1 = 1.959;
k2 = 0.709;
k3 = 1.09;
n = 1:10000;
len = length(n);
R1 = zeros(1,length(n));
R2 = zeros(1,length(n));
R3 = zeros(1,length(n));
for i = 1:len
    R1(i) = Rn(k1,l1,n(i));
    R2(i) = Rn(k2,l2,n(i));
    R3(i) = Rn(k3,l3,n(i));
end
%}
%plot(n,R1,'r',n,R2,'k',n,R3,'b')

%p = parpool(4);
%{
lambda = ones(1,100)*4.5;
epsilon1 = zeros(1,length(lambda));
U1 = zeros(1,length(lambda));
for i=1:length(lambda)
    [U1(i),epsilon1(i) ,~] = ProspectTheoryBased( lambda(i),1,1,5000,50,0,2,0,0.002:0.0001:0.02);
end
%plot(lambda,Opt_epsilon_max_1);
%U_mean = [U_mean mean(U1)]
mean(U1)
mean(epsilon1)
%}
%opt_eps_max_appro =[0.0132    0.0105    0.0084    0.0066    0.0062];
%{
N=[5000 10000 20000 30000 40000];
opt_eps_max_appro = zeros(1,length(N));
opt_eps_max1 = zeros(1,length(N));
opt_eps_max2 = zeros(1,length(N));
opt_eps_max3 = zeros(1,length(N));
U_cmp = zeros(1,length(N));
for i=1:length(N)
    [ ~,opt_eps_max1(i),~] = ProspectTheoryBased( 2.25,1,1,N(i),1,0.5,0.5,0,0.001:0.0001:0.03 );
    [ ~,opt_eps_max2(i),~] = ProspectTheoryBased( 2.25,0.97,0.97,N(i),1,0.5,0.5,0,0.001:0.0001:0.03 );
    [ ~,opt_eps_max3(i),~] = ProspectTheoryBased( 2.25,0.88,0.88,N(i),1,0.5,0.5,0,0.001:0.0001:0.03 );
    [ opt_eps_max_appro(i) ] = Approximation_fun( 2.25,N(i),1,0.5,0.5 );
end
plot(N,opt_eps_max1,'*-',N,opt_eps_max2,'x-',N,opt_eps_max3,'^-',N,opt_eps_max_appro,'o-','LineWidth',2,'Markers',10);
%}
%legend('Without approximation','With approximation');
%xlabel('N','FontSize',15);
%ylabel('optmal \epsilon^*','FontSize',15);
%set(gca,'FontSize',12);
%dif = opt_eps_max-opt_eps_max_appro;
%per = dif./opt_eps_max

%{
N=[5000 10000 20000 20000 40000 50000];
U_cmp = zeros(1,length(N));
for i=1:length(N)
    [ U_cmp(i),~,~] = ProspectTheoryBased( 2.25,1,1,N(i),1,0.5,0.5,0,opt_eps_max_appro(i));
    %[ opt_eps_max_appro(i) ] = Approximation(  2.25,N(i),1,0.5,0.5 );
end
%plot(N,opt_eps_max,'*',N,opt_eps_max_appro,'o');
U_dif = U-U_cmp;
per = U_dif./U
%}
%saveresult([Opt_epsilon_max_1,Opt_U_1],'E:\liao\MATLAB\Differential Privacy\Data\','data_1','.txt');


%for i=2:length(lambda)
%    [ Opt_U_2(i),Opt_epsilon_max_2(i),~] = ProspectTheoryBased( lambda(i),0.88,0.88,5000,1,0,2,0.005,0.002:0.0001:0.02 );
%end
%saveresult([Opt_epsilon_max_2,Opt_U_2],'E:\liao\MATLAB\Differential Privacy\Data\','data_2','.txt');
%{
Opt_U_3=zeros(6,1);
Opt_epsilon_max_3 = zeros(6,1);
for i=1:length(lambda)
    [ Opt_U_3(i),Opt_epsilon_max_3(i),~] = ProspectTheoryBased( lambda(i),0.88,0.88,5000,1,0,2,0.03,0.002:0.0001:0.025 );
end
saveresult([Opt_epsilon_max_3,Opt_U_3],'E:\liao\MATLAB\Differential Privacy\Data\','data_3','.txt');
%{
lambda = 1:0.5:4.5;
Opt_U_1=zeros(1,length(lambda));
Opt_epsilon_max_1 = zeros(1,length(lambda));
for i=1:length(lambda)
    [ Opt_U_1(i),Opt_epsilon_max_1(i),~] = ProspectTheoryBased( lambda(i),0.88,0.88,5000,1,0,2,0,0.002:0.0001:0.025 );
end
%}

[ ~,Opt_epsilon_EUT_max,~] = ProspectTheoryBased( 1,1,1,10000,50,0.5,0.5,0,0.0001:0.0001:0.01 );

Uc_byEUT=zeros(1,length(lambda));
for i=1:length(lambda)
    [ Uc_byEUT(i),~,~] = ProspectTheoryBased( lambda(i),0.88,0.88,10000,50,0.5,0.5,0,Opt_epsilon_EUT_max);
end

plot(lambda,Opt_U,'*',lambda,Uc_byEUT,'o');
dif=Opt_U-Uc_byEUT;
per = dif./Opt_U*100    
%}
%{
Opt_epsilon_max_1=[0.0236,0.0207,0.0185,0.0170,0.0160,0.0151];
%ProspectTheoryBased( lambda(i),0.88,0.88,5000,1,0,2,0.01,0.01:0.0001:0.02
%);lambda 1:0.5:3.5
Opt_epsilon_max_2=[0.0224,0.0203,0.0188,0.0176,0.0167,0.0161];
%ProspectTheoryBased( lambda(i),0.88,0.88,5000,1,0,2,0.04,0.02:0.0001:0.04 );lambda 1:0.5:3.5
Opt_epsilon_max_3=[0.0243,0.0243,0.0243,0.0243,0.0243,0.0243];
lambda = 1:0.5:3.5;
plot(lambda,Opt_epsilon_max_1,'b*-',lambda,Opt_epsilon_max_2,'ro-.',lambda,Opt_epsilon_max_3,'k^--','LineWidth',2);
xlabel('\lambda');
ylabel('Optimal \epsilon');
set(gca,'FontSize',15);
h = legend('\epsilon_{ref}=0','\epsilon_{ref}=0.01','\epsilon_{ref}=0.04')
set(h,'FontSize',20)

%}

%[U,epsilon ,~] = ProspectTheoryBased( 1,0.88,0.88,5000,1,0,2,0.02,0.002:0.0001:0.03)
%{
lambda_1 = 2.25;
beta_1 = 0.88;
alpha_1 = 0.88;
N_1 = 1000;
u_1 = 0.65;
epsilon_ref_1 = 0 ;

lambda_2 = 4;
beta_2 = 0.5;
alpha_2 = 0.88;
N_2 = 1000;
u_2 = 0.65;
epsilon_ref_2 = 0.001 ;

C = 1;
Wm = 0.5;
Wl = 0.5;
Epsilon_searchrange = 0.002:0.0001:0.05;
data_max = 1;
le = 1;


average=0.5;
sigma=1;
lower=0;
upper=inf;
iteration=100;
%}

%{
N_2 = 200:200:2000;
Opt_epsilon_max_1 = zeros(1,length(N_2));
Opt_epsilon_max_2 = zeros(1,length(N_2));
Opt_epsilon_max_3 = zeros(1,length(N_2));
for i = 1:length(N_2)
    
    [ Opt_U_1,Opt_epsilon_max_1(i), Opt_num_1] = ProspectTheoryBased( lambda_2,beta_2,alpha_2,u_2,epsilon_ref_2,N_1+N_2(i),C,Wm,Wl,Epsilon_searchrange );
    
    [ Opt_U_2,Opt_epsilon_max_2(i), Opt_num_2 ] = ProspectTheoryBase_TwoGroups( lambda_1,beta_1,alpha_1,u_1,epsilon_ref_1,N_1,lambda_2,beta_2,alpha_2,u_2,N_2(i),epsilon_ref_2,C,Wm,Wl,Epsilon_searchrange,data_max,le);

    [ Opt_U_3,Opt_epsilon_max_3(i), Opt_num_3] = ProspectTheoryBased( lambda_1,beta_1,alpha_1,u_1,epsilon_ref_1,N_1+N_2(i),C,Wm,Wl,Epsilon_searchrange );
end
%}
%{
beta_2 = 0.4:0.1:0.9;
Opt_epsilon_max_1 = zeros(1,length(beta_2));

for i = 1:length(beta_2)
    
    [ Opt_U_1,Opt_epsilon_max_1(i), Opt_num_1] = ProspectTheoryBased( lambda_2,beta_2(i),alpha_2,u_2,epsilon_ref_2,N_1+N_2,C,Wm,Wl,Epsilon_searchrange );
  
end
%}
%plot(beta_2,Opt_epsilon_max_1,'-ro')
%{
lambda_1 = 1.5:0.5:3.5;
Opt_epsilon_max_1 = zeros(1,length(lambda_1));
Opt_epsilon_max_2 = zeros(1,length(lambda_1));
Opt_epsilon_max_3 = zeros(1,length(lambda_1));
tic;

for i = 1:length(lambda_1)
    
%{
    beta_1 = 0.4;
    [ Opt_U_1,Opt_epsilon_max_1(i), Opt_num_1] = ProspectTheoryBased_Uniform( lambda_1(i),beta_1,alpha_1,u_1,epsilon_ref_1,N_1+N_2,C,Wm,Wl,Epsilon_searchrange);
    beta_1 = 0.6;
    [ Opt_U_2,Opt_epsilon_max_2(i), Opt_num_2] = ProspectTheoryBased_Uniform( lambda_1(i),beta_1,alpha_1,u_1,epsilon_ref_1,N_1+N_2,C,Wm,Wl,Epsilon_searchrange);
    beta_1 = 0.8;
    [ Opt_U_3,Opt_epsilon_max_3(i), Opt_num_3] = ProspectTheoryBased_Uniform( lambda_1(i),beta_1,alpha_1,u_1,epsilon_ref_1,N_1+N_2,C,Wm,Wl,Epsilon_searchrange);
  %}  
    beta_1 = 0.4;
    [ Opt_U_1,Opt_epsilon_max_1(i), Opt_num_1] = ProspectTheoryBased_TruncatedNormal( lambda_1(i),beta_1,alpha_1,u_1,epsilon_ref_1,N_1+N_2,C,average,sigma,lower,upper,iteration,Epsilon_searchrange);
    beta_1 = 0.6;
    [ Opt_U_2,Opt_epsilon_max_2(i), Opt_num_2] = ProspectTheoryBased_TruncatedNormal( lambda_1(i),beta_1,alpha_1,u_1,epsilon_ref_1,N_1+N_2,C,average,sigma,lower,upper,iteration,Epsilon_searchrange);
    beta_1 = 0.8;
    [ Opt_U_3,Opt_epsilon_max_3(i), Opt_num_3] = ProspectTheoryBased_TruncatedNormal( lambda_1(i),beta_1,alpha_1,u_1,epsilon_ref_1,N_1+N_2,C,average,sigma,lower,upper,iteration,Epsilon_searchrange);
  
end
toc

plot(lambda_1,Opt_epsilon_max_1,'-ro',lambda_1,Opt_epsilon_max_2,'-bs',lambda_1,Opt_epsilon_max_3,'-k^','LineWidth',2);
%}


%{
Lambda_spec.ave=[2 5];
Lambda_spec.sigma=[1 1];
Lambda_spec.probability = [0.3 0.7];
Lambda_spec.lower = [1 1];
Lambda_spec.upper = [inf inf];
%Lambda_spec.N = 2000;

Beta_spec.ave=0.5;
Beta_spec.sigma=0.1;
Beta_spec.probability = 1;
Beta_spec.lower = 0;
Beta_spec.upper = 1;
%Beta_spec.N = 2000;
Alpha = 0.5;
Mu = 0.8;

Ref = 0;
Num = 2000;

C=1;
average=0.5;
sigma=1;
lower=0;
upper=inf;
iteration=30;
Epsilon_searchrange = 0.002:0.0002:0.03;

[ Opt_U,Opt_epsilon_max, Opt_num] = ProspectTheoryBased_TruncatedNormal_GMPTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,Num,C,average,sigma,lower,upper,iteration,Epsilon_searchrange )
%}





%{
ave=[3 4;2.5 4.5;2 5;1.5 5.5];
Lambda_spec.ave = [3 4];
Lambda_spec.sigma=[0.1 0.1];
Lambda_spec.probability = [0.5 0.5];
Lambda_spec.lower = [1 1];
Lambda_spec.upper = [inf inf];
%Lambda_spec.N = 2000;

Beta_spec.ave=0.5;
Beta_spec.sigma=0.1;
Beta_spec.probability = 1;
Beta_spec.lower = 0;
Beta_spec.upper = 1;
%Beta_spec.N = 2000;
Alpha = 0.5;
Mu = 0.8;

Ref = 0;
Num = 5000;

C=1;
W = 1;
average=0.5;
sigma=1;
lower=0;
upper=inf;
Epsilon_searchrange = 0.0002:0.0002:0.02;


lambda = Lambda_spec.ave*Lambda_spec.probability';
beta = Beta_spec.ave*Beta_spec.probability;
alpha = beta;
Mu = 0.8;
epsilon_ref = 0;

iteration=20;
%{
tic;
[ Opt_U,Opt_epsilon_max, Opt_num,~] = ProspectTheoryBased_GMPTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);
toc
%}



%Lambda_spec.ave = 3.5;
%Lambda_spec.sigma=0;
%Lambda_spec.probability = 1;
%Lambda_spec.lower = 1;
%Lambda_spec.upper = inf;


%[ Opt_U,Opt_epsilon_max, Opt_num,~] = ProspectTheoryBased_GMPTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);
%[ Opt_U_1,Opt_epsilon_max_1, Opt_num_1] = ProspectTheoryBased_TruncatedNormal( lambda,beta,alpha,Mu,epsilon_ref,Num,C,average,sigma,lower,upper,iteration,Epsilon_searchrange);

%{
Beta_spec.ave = 0.5;
Beta_spec.sigma=0;
Beta_spec.probability = 1;
Beta_spec.lower = 0;
Beta_spec.upper = 1;


[ Opt_U_2,Opt_epsilon_max_2, Opt_num_2,~] = ProspectTheoryBased_GMPTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);
%}

%[ Opt_U_1,Opt_epsilon_max_1, Opt_num_1] = ProspectTheoryBased_TruncatedNormal( lambda,beta,alpha,Mu,epsilon_ref,Num,C,average,sigma,lower,upper,iteration,Epsilon_searchrange);



Opt_U = zeros(1,length(ave));
Opt_epsilon_max = zeros(1,length(ave));
Opt_num = zeros(1,length(ave));
U = zeros(1,length(ave));

for i = 1: size(ave,1)
    tic;
    Lambda_spec.ave = ave(i,:);
    [ Opt_U(i),Opt_epsilon_max(i), Opt_num(i),U(i)] = ProspectTheoryBased_GMPTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);
    %[ Opt_U(i),Opt_epsilon_max(i), Opt_num(i),U(i)] = ProspectTheoryBased_TruncatedNormal_GMPTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,Num,C,average,sigma,lower,upper,iteration,Epsilon_searchrange );
    toc
end
%}


%{
Opt_U_3 = zeros(1,length(ave));
Opt_epsilon_max_3 = zeros(1,length(ave));
Opt_num_3= zeros(1,length(ave));
U_3 = zeros(1,length(ave));

beta_sigma = 0.1:0.2:0.7;

for i = 1: length(beta_sigma)
    tic;
    Beta_spec.sigma = beta_sigma(i);
    [ Opt_U_3(i),Opt_epsilon_max_3(i), Opt_num_3(i),U_3(i)] = ProspectTheoryBased_GMPTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);
    %[ Opt_U(i),Opt_epsilon_max(i), Opt_num(i),U(i)] = ProspectTheoryBased_TruncatedNormal_GMPTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,Num,C,average,sigma,lower,upper,iteration,Epsilon_searchrange );
    toc
end
%}
PT_parameters1_gamma_fit;
lambda_spec.distribution = 'Gamma';
lambda_spec.shape = par_gam_lambda(1);
lambda_spec.scale = par_gam_lambda(2);
%lambda_spec.N = 2000;

lambda_homo_spec.distribution = 'Homogeneity';
lambda_homo_spec.value = par_gam_lambda(1)*par_gam_lambda(2);
%lambda_homo_spec.N = 2000;

lambda_uniform_spec.distribution = 'Uniform';
lambda_uniform_spec.mean = par_gam_lambda(1)*par_gam_lambda(2);
lambda_uniform_spec.range = par_gam_lambda(1)*par_gam_lambda(2)*2;

beta_spec.distribution = 'Gamma';
beta_spec.shape = par_gam_beta(1);
beta_spec.scale = par_gam_beta(2);
%beta_spec.N = 2000;

beta_homo_spec.distribution = 'Homogeneity';
beta_homo_spec.value = par_gam_beta(1)*par_gam_beta(2);
%beta_homo_spec.N = 2000;

beta_uniform_spec.distribution = 'Uniform';
beta_uniform_spec.mean = par_gam_beta(1)*par_gam_beta(2);
beta_uniform_spec.range = par_gam_beta(1)*par_gam_beta(2)*2;

W_homo_spec.distribution = 'Homogeneity';
W_homo_spec.value = 0.03;


W_spec.distribution = 'TruncatedNormal';
W_spec.average = 0.25;
W_spec.sigma = 0.5;
W_spec.lower = 0;
W_spec.upper = 0.5;

Mu = 0.8;
Alpha = 0.88;
Ref = 0;

Num = 2000;
C=1;
W = 0.1;
Epsilon_searchrange = 0.002:0.0001:0.02;

%{
iteration = 20;
Ref = 0.01;
Num = 5000;
C = 1;
Epsilon_searchrange = 0.002:0.0001:0.02;

lambda_homo_spec.value = 1.5;
W_spec.average = 1;
W_spec.sigma = 1;
W_spec.lower = 0;
W_spec.upper = 2;

beta = 0.4:0.1:0.9;
Opt_U_1 = zeros(1,length(beta));
Opt_epsilon_max_1 = zeros(1,length(beta));
Opt_num_1= zeros(1,length(beta));

for i = 1:length(beta)
    tic;
    beta_homo_spec.value = beta(i);
    
    [ Opt_U_1(i),Opt_epsilon_max_1(i), Opt_num_1(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end

plot(beta,Opt_epsilon_max_1)


iteration = 20;
Ref = 0.01;
Num = 5000;
C = 1;
Epsilon_searchrange = 0.002:0.0001:0.02;

lambda_homo_spec.value = 1.5;
W_spec.average = 1;
W_spec.sigma = 1;
W_spec.lower = 0;
W_spec.upper = 2;

beta = 0.4:0.1:0.9;
Opt_U_1 = zeros(1,length(beta));
Opt_epsilon_max_1 = zeros(1,length(beta));
Opt_num_1= zeros(1,length(beta));

for i = 1:length(beta)
    tic;
    beta_homo_spec.value = beta(i);
    
    [ Opt_U_1(i),Opt_epsilon_max_1(i), Opt_num_1(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end

plot(beta,Opt_epsilon_max_1)
%}

%{
iteration = 30;
Ref = 0.01;
Num = 5000;
C = 1;
Epsilon_searchrange = 0.005:0.0001:0.02;

lambda_homo_spec.value = 2.5;
W_spec.average = 1;
W_spec.sigma = 1;
W_spec.lower = 0;
W_spec.upper = 2;

beta = 0.4:0.1:0.9;
Opt_U_3 = zeros(1,length(beta));
Opt_epsilon_max_3 = zeros(1,length(beta));
Opt_num_3= zeros(1,length(beta));

for i = 1:length(beta)
    tic;
    beta_homo_spec.value = beta(i);
    
    [ Opt_U_3(i),Opt_epsilon_max_3(i), Opt_num_3(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end

plot(beta,Opt_epsilon_max_3,'ro-')


iteration = 40;
Ref = 0;
Num = 5000;
C = 1;
Epsilon_searchrange = 0.005:0.0001:0.02;

lambda_homo_spec.value = 2.5;
W_spec.average = 1;
W_spec.sigma = 1;
W_spec.lower = 0;
W_spec.upper = 2;

beta = 0.4:0.1:0.8;
Opt_U_4 = zeros(1,length(beta));
Opt_epsilon_max_4 = zeros(1,length(beta));
Opt_num_4 = zeros(1,length(beta));

for i = 1:length(beta)
    tic;
    beta_homo_spec.value = beta(i);
    [ Opt_U_4(i),Opt_epsilon_max_4(i), Opt_num_4(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end

hold on;
plot(beta,Opt_epsilon_max_4,'b*-')
%}
%{
iteration = 30;
Ref = 0.01;
Num = 5000;
C = 1;
Epsilon_searchrange = 0.005:0.0001:0.02;

lambda_homo_spec.value = 3;
W_spec.average = 1;
W_spec.sigma = 1;
W_spec.lower = 0;
W_spec.upper = 2;

beta = 0.4:0.1:0.9;
Opt_U_5 = zeros(1,length(beta));
Opt_epsilon_max_5 = zeros(1,length(beta));
Opt_num_5= zeros(1,length(beta));

for i = 1:length(beta)
    tic;
    beta_homo_spec.value = beta(i);
    
    [ Opt_U_5(i),Opt_epsilon_max_5(i), Opt_num_5(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end

plot(beta,Opt_epsilon_max_5,'ro-')


iteration = 30;
Ref = 0;
Num = 5000;
C = 1;
Epsilon_searchrange = 0.005:0.0001:0.02;

lambda_homo_spec.value = 3;
W_spec.average = 1;
W_spec.sigma = 1;
W_spec.lower = 0;
W_spec.upper = 2;

beta = 0.4:0.1:0.9;
Opt_U_6 = zeros(1,length(beta));
Opt_epsilon_max_6 = zeros(1,length(beta));
Opt_num_6 = zeros(1,length(beta));

for i = 1:length(beta)
    tic;
    beta_homo_spec.value = beta(i);
    [ Opt_U_6(i),Opt_epsilon_max_6(i), Opt_num_6(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end

hold on;
plot(beta,Opt_epsilon_max_6,'b*-')
%}

iteration = 50;
Ref = 0;
Num = 5000;
C = 1;
Epsilon_searchrange = 0.005:0.0001:0.020;

W_spec.average = 1;
W_spec.sigma = 1;
W_spec.lower = 0;
W_spec.upper = 2;

lambda = 1.5:0.5:3.5;
beta = 0.4:0.1:0.8;

%{
Opt_U_3 = zeros(length(beta),length(lambda));
Opt_epsilon_max_3 = zeros(length(beta),length(lambda));
Opt_num_3 = zeros(length(beta),length(lambda));

Opt_U_4 = zeros(length(beta),length(lambda));
Opt_epsilon_max_4 = zeros(length(beta),length(lambda));
Opt_num_4 = zeros(length(beta),length(lambda));
%}

for i = 3
    for j = 4
        tic;
        Ref = 0;
        beta_homo_spec.value = beta(j);
        Alpha = beta(j);
        lambda_homo_spec.value = lambda(i);
        
        [ Opt_U_3(j,i),Opt_epsilon_max_5(j,i), Opt_num_3(j,i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
        
        Ref=0.01;
        [ Opt_U_4(j,i),Opt_epsilon_max_6(j,i), Opt_num_4(j,i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
        toc
    end
end

%plot(lambda,Opt_epsilon_max_3(3,:),'-b*',lambda,Opt_epsilon_max_4(3,:),'-.ro','LineWidth',2,'MarkerSize',8)

%{
Opt_U_1 = zeros(1,length(lambda));
Opt_epsilon_max_1 = zeros(1,length(lambda));
Opt_num_1 = zeros(1,length(lambda));


Opt_U_2 = zeros(1,length(lambda));
Opt_epsilon_max_2 = zeros(1,length(lambda));
Opt_num_2 = zeros(1,length(lambda));
%}
%{

Opt_U_9 = zeros(1,length(lambda));
Opt_epsilon_max_9 = zeros(1,length(lambda));
Opt_num_9 = zeros(1,length(lambda));
%}
%{
for i = 1:length(lambda)
    tic;
    beta_homo_spec.value = 0.4;
    lambda_homo_spec.value = lambda(i);
    [ Opt_U_7(i),Opt_epsilon_max_7(i), Opt_num_7(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end

for i = 1:length(lambda)
    tic;
    beta_homo_spec.value = 0.6;
    lambda_homo_spec.value = lambda(i);
    [ Opt_U_8(i),Opt_epsilon_max_8(i), Opt_num_8(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
for i = 4:5
    tic;
    Ref = 0;
    beta_homo_spec.value = 0.88;
    lambda_homo_spec.value = lambda(i);
    [ Opt_U_1(i),Opt_epsilon_max_1(i), Opt_num_1(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
toc
end



for i = 6
    tic;
    Ref = 0.01;
    beta_homo_spec.value = 0.88;
    lambda_homo_spec.value = lambda(i);
    [ Opt_U_2(i),Opt_epsilon_max_2(i), Opt_num_2(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
toc
end

plot(lambda,Opt_epsilon_max_1,'-b*',lambda,Opt_epsilon_max_2,'-.ro','LineWidth',2,'MarkerSize',8)
%}
%plot(lambda,Opt_epsilon_max_7,'--k^',lambda,Opt_epsilon_max_8,'-.ro',lambda,Opt_epsilon_max_9,'-b*','LineWidth',2,'MarkerSize',8)

%{
iteration = 40;
Ref = 0.04;
Num = 5000;
C = 1;
Epsilon_searchrange = 0.005:0.0001:0.02;

lambda_homo_spec.value = 3;
W_spec.average = 1;
W_spec.sigma = 1;
W_spec.lower = 0;
W_spec.upper = 2;

beta = 0.4:0.1:0.9;
Opt_U_7 = zeros(1,length(beta));
Opt_epsilon_max_7 = zeros(1,length(beta));
Opt_num_7 = zeros(1,length(beta));

for i = 1:length(beta)
    tic;
    beta_homo_spec.value = beta(i);
    [ Opt_U_7(i),Opt_epsilon_max_7(i), Opt_num_7(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);
    toc
end

hold on;
plot(beta,Opt_epsilon_max_7,'ks-')
%}


%{
range_lambda = 0.2:0.2:1;
Opt_U = zeros(1,length(range_lambda));
Opt_epsilon_max = zeros(1,length(range_lambda));
Opt_num= zeros(1,length(range_lambda));
U = zeros(1,length(range_lambda));

for i = 1:length(range_lambda)
    tic;
    lambda_uniform_spec.range=range_lambda(i);
    [ Opt_U(i),Opt_epsilon_max(i), Opt_num(i),U(i)] = ProspectTheoryBased_PTPara( lambda_uniform_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end

plot(range_lambda,Opt_epsilon_max);
%}
%{
range_lambda_2 = 1:0.5:3.5;
Opt_U_2 = zeros(1,length(range_lambda_2));
Opt_epsilon_max_2 = zeros(1,length(range_lambda_2));
Opt_num_2= zeros(1,length(range_lambda_2));
U = zeros(1,length(range_lambda_2));

for i = 1:length(range_lambda_2)
    tic;
    lambda_uniform_spec.range=range_lambda_2(i);
    [ Opt_U_2(i),Opt_epsilon_max_2(i), Opt_num_2(i),U(i)] = ProspectTheoryBased_PTPara( lambda_uniform_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end

plot(range_lambda_2,Opt_epsilon_max_2);
%}
%{
W_homo_spec.value = 0.05;

range_lambda_3 = 0.2:0.2:1;
Opt_U_3 = zeros(1,length(range_lambda_3));
Opt_epsilon_max_3 = zeros(1,length(range_lambda_3));
Opt_num_3= zeros(1,length(range_lambda_3));
U = zeros(1,length(range_lambda_3));

for i = 1:length(range_lambda_3)
    tic;
    lambda_uniform_spec.range=range_lambda_3(i);
    [ Opt_U_3(i),Opt_epsilon_max_3(i), Opt_num_3(i),U(i)] = ProspectTheoryBased_PTPara( lambda_uniform_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
plot(range_lambda_3,Opt_epsilon_max_3);
%}
%{
W_homo_spec.value = 0.05;
%lambda_uniform_spec.mean = 2.25;

range_lambda_4 = 1:0.5:3.5;
Opt_U_4 = zeros(1,length(range_lambda_4));
Opt_epsilon_max_4 = zeros(1,length(range_lambda_4));
Opt_num_4= zeros(1,length(range_lambda_4));
U = zeros(1,length(range_lambda_4));

for i = 1:length(range_lambda_4)
    tic;
    lambda_uniform_spec.range=range_lambda_4(i);
    [ Opt_U_4(i),Opt_epsilon_max_4(i), Opt_num_4(i),U(i)] = ProspectTheoryBased_PTPara( lambda_uniform_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
plot(range_lambda_4,Opt_epsilon_max_4);
%}
%{
Epsilon_searchrange = 0.0001:0.00005:0.0015;
iteration = 30;
Num = 2000;
C=2.5;
W_homo_spec.value = 0.02;

lambda_uniform_spec.mean = 1.95;
range_lambda_5 = 0.2:0.2:3;
Opt_U_5 = zeros(1,length(range_lambda_5));
Opt_epsilon_max_5 = zeros(1,length(range_lambda_5));
Opt_num_5= zeros(1,length(range_lambda_5));
U = zeros(1,length(range_lambda_5));

for i = 1:length(range_lambda_5)
    tic;
    lambda_uniform_spec.range=range_lambda_5(i);
    [ Opt_U_5(i),Opt_epsilon_max_5(i), Opt_num_5(i)] = ProspectTheoryBased_PTPara( lambda_uniform_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
plot(range_lambda_5/2,Opt_epsilon_max_5);
%}
%{

Epsilon_searchrange = 0.0001:0.00005:0.0015;
iteration = 30;
Num = 2000;
C=2.5;
W_homo_spec.value = 0.02;
lambda_uniform_spec.mean = 1.95;
range_lambda_5 = 0.2:0.2:3;
Opt_U_5 = zeros(1,length(range_lambda_5));
Opt_epsilon_max_5 = zeros(1,length(range_lambda_5));
Opt_num_5= zeros(1,length(range_lambda_5));
U = zeros(1,length(range_lambda_5));

for i = 1:length(range_lambda_5)
    tic;
    lambda_uniform_spec.range=range_lambda_5(i);
    [ Opt_U_5(i),Opt_epsilon_max_5(i), Opt_num_5(i)] = ProspectTheoryBased_PTPara( lambda_uniform_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
plot(range_lambda_5/2,Opt_epsilon_max_5);
%}

%{
Epsilon_searchrange = 0.0001:0.00005:0.002;
iteration = 30;
Num = 6000;
C=2.5;
W_homo_spec.value = 0.02;
lambda_uniform_spec.mean = 1.95;
range_lambda_6 = 0.2:0.2:3.6;
Opt_U_6 = zeros(1,length(range_lambda_6));
Opt_epsilon_max_6 = zeros(1,length(range_lambda_6));
Opt_num_6= zeros(1,length(range_lambda_6));
U = zeros(1,length(range_lambda_6));

for i = 1:length(range_lambda_6)
    tic;
    lambda_uniform_spec.range=range_lambda_6(i);
    [ Opt_U_6(i),Opt_epsilon_max_6(i), Opt_num_6(i)] = ProspectTheoryBased_PTPara( lambda_uniform_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
plot(range_lambda_6,Opt_epsilon_max_6);
%}
%{
Epsilon_searchrange = 0.0005:0.0001:0.004;
iteration = 30;
Num = 2000;
C=2.5;
W_homo_spec.value = 0.02;
beta_uniform_spec.mean = 0.75;
range_beta_7 = 0.1:0.1:0.5;
Opt_U_7 = zeros(1,length(range_beta_7));
Opt_epsilon_max_7 = zeros(1,length(range_beta_7));
Opt_num_7= zeros(1,length(range_beta_7));

for i = 1:length(range_beta_7)
    tic;
    beta_uniform_spec.range=range_beta_7(i);
    [ Opt_U_7(i),Opt_epsilon_max_7(i), Opt_num_7(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_uniform_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
plot(range_beta_7,Opt_epsilon_max_7);
%}


%{
%2018.3.26 
Epsilon_searchrange = 0.0005:0.0001:0.0015;
iteration = 100;
Num = 2000;
C=2.5;
W_homo_spec.value = 0.02;

scale_beta_8 = 0.0001:0.0005:0.008;
Opt_U_8 = zeros(1,length(scale_beta_8));
Opt_epsilon_max_8 = zeros(1,length(scale_beta_8));
Opt_num_8= zeros(1,length(scale_beta_8));

for i = 1:length(scale_beta_8)
    tic;
    beta_spec.scale=scale_beta_8(i);
    beta_spec.shape = par_gam_beta(1)*par_gam_beta(2)/beta_spec.scale;
    [ Opt_U_8(i),Opt_epsilon_max_8(i), Opt_num_8(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end

plot(scale_beta_8*par_gam_beta(1)*par_gam_beta(2),Opt_epsilon_max_8)
%}

%plot(range_beta_8,Opt_epsilon_max_8);

%[ Opt_U_1,Opt_epsilon_max_1, Opt_num_1,U_1] = ProspectTheoryBased_PTPara( lambda_spec,beta_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);

%[ Opt_U_2,Opt_epsilon_max_2, Opt_num_2,U_2] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_spec,iteration,Epsilon_searchrange);



%{
scale_lambda = 0.1:0.1:0.9;
Opt_U_3 = zeros(1,length(scale_lambda));
Opt_epsilon_max_3 = zeros(1,length(scale_lambda));
Opt_num_3= zeros(1,length(scale_lambda));
U_3 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_3(i),Opt_epsilon_max_3(i), Opt_num_3(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}

%{
scale_lambda = 0.4:0.1:0.7;
Opt_U_4 = zeros(1,length(scale_lambda));
Opt_epsilon_max_4 = zeros(1,length(scale_lambda));
Opt_num_4= zeros(1,length(scale_lambda));
U_4 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_4(i),Opt_epsilon_max_4(i), Opt_num_4(i),U_4(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
scale_lambda = 0.4:0.1:0.7;
Opt_U_5 = zeros(1,length(scale_lambda));
Opt_epsilon_max_5 = zeros(1,length(scale_lambda));
Opt_num_5= zeros(1,length(scale_lambda));
U_5 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_5(i),Opt_epsilon_max_5(i), Opt_num_5(i),U_5(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
%2018.3.27
Epsilon_searchrange = 0.002:0.0001:0.006;
iteration = 60;
Num = 2000;
C=1;
W_homo_spec.value = 0.03;

scale_lambda = 0.1:0.1:1.5;
Opt_U_66 = zeros(1,length(scale_lambda));
Opt_epsilon_max_66 = zeros(1,length(scale_lambda));
Opt_num_66= zeros(1,length(scale_lambda));
%U_6 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_66(i),Opt_epsilon_max_66(i), Opt_num_66(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
plot(scale_lambda*par_gam_lambda(1)*par_gam_lambda(2),Opt_epsilon_max_66)
xlabel('Var(\lambda)')
ylabel('Optimal \epsilon^*')
%}
%{
scale_lambda = 0.3:0.3:1.5;
Opt_U_7 = zeros(1,length(scale_lambda));
Opt_epsilon_max_7 = zeros(1,length(scale_lambda));
Opt_num_7= zeros(1,length(scale_lambda));
U_7 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_7(i),Opt_epsilon_max_7(i), Opt_num_7(i),U_7(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
scale_lambda = 0.3:0.3:1.5;
Opt_U_8 = zeros(1,length(scale_lambda));
Opt_epsilon_max_8 = zeros(1,length(scale_lambda));
Opt_num_8= zeros(1,length(scale_lambda));
U_8 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_8(i),Opt_epsilon_max_8(i), Opt_num_8(i),U_8(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
scale_lambda = 0.1:0.1:1.0;
Opt_U_9 = zeros(1,length(scale_lambda));
Opt_epsilon_max_9 = zeros(1,length(scale_lambda));
Opt_num_9= zeros(1,length(scale_lambda));
U_9 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_9(i),Opt_epsilon_max_9(i), Opt_num_9(i),U_9(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
scale_lambda = 1.0:0.1:2.0;
Opt_U_10 = zeros(1,length(scale_lambda));
Opt_epsilon_max_10 = zeros(1,length(scale_lambda));
Opt_num_10= zeros(1,length(scale_lambda));
U_10 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_10(i),Opt_epsilon_max_10(i), Opt_num_10(i),U_10(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
scale_lambda = 0.1:0.1:2.0;
Opt_U_12 = zeros(1,length(scale_lambda));
Opt_epsilon_max_12 = zeros(1,length(scale_lambda));
Opt_num_12= zeros(1,length(scale_lambda));
U_12 = zeros(1,length(scale_lambda));

for i = 1:length(scale_lambda)
    tic;
    lambda_spec.scale = scale_lambda(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda(i);
    [ Opt_U_12(i),Opt_epsilon_max_12(i), Opt_num_12(i),U_12(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
Num = 10000;
C = 0.1;
scale_lambda_14 = 0.3:0.3:1.5;
Opt_U_14 = zeros(1,length(scale_lambda_14));
Opt_epsilon_max_14 = zeros(1,length(scale_lambda_14));
Opt_num_14= zeros(1,length(scale_lambda_14));
U_14 = zeros(1,length(scale_lambda_14));

for i = 1:length(scale_lambda_14)
    tic;
    lambda_spec.scale = scale_lambda_14(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda_14(i);
    [ Opt_U_14(i),Opt_epsilon_max_14(i), Opt_num_14(i),U_14(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
Num = 600;
C = 1;
scale_lambda_15 = 0.3:0.3:1.5;
Opt_U_15 = zeros(1,length(scale_lambda_15));
Opt_epsilon_max_15 = zeros(1,length(scale_lambda_15));
Opt_num_15= zeros(1,length(scale_lambda_15));
U_15 = zeros(1,length(scale_lambda_15));

for i = 1:length(scale_lambda_15)
    tic;
    lambda_spec.scale = scale_lambda_15(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda_15(i);
    [ Opt_U_15(i),Opt_epsilon_max_15(i), Opt_num_15(i),U_15(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
Num = 1000;
C = 1;
scale_lambda_16 = 0.3:0.3:1.5;
Opt_U_16 = zeros(1,length(scale_lambda_16));
Opt_epsilon_max_16 = zeros(1,length(scale_lambda_16));
Opt_num_16= zeros(1,length(scale_lambda_16));
U_16 = zeros(1,length(scale_lambda_16));

for i = 1:length(scale_lambda_16)
    tic;
    lambda_spec.scale = scale_lambda_16(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda_16(i);
    [ Opt_U_16(i),Opt_epsilon_max_16(i), Opt_num_16(i),U_16(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}
%{
Num = 1000;
C = 1;
scale_lambda_17 = 0.05:0.05:0.25;
Opt_U_17 = zeros(1,length(scale_lambda_17));
Opt_epsilon_max_17 = zeros(1,length(scale_lambda_17));
Opt_num_17= zeros(1,length(scale_lambda_17));
U_17 = zeros(1,length(scale_lambda_17));

for i = 1:length(scale_lambda_17)
    tic;
    lambda_spec.scale = scale_lambda_17(i);
    lambda_spec.shape = par_gam_lambda(1)*par_gam_lambda(2)/scale_lambda_17(i);
    [ Opt_U_17(i),Opt_epsilon_max_17(i), Opt_num_17(i),U_17(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}



%{
scale_beta = 0.04:0.02:0.10;
Opt_U_4 = zeros(1,length(scale_beta));
Opt_epsilon_max_4 = zeros(1,length(scale_beta));
Opt_num_4= zeros(1,length(scale_beta));
U_4 = zeros(1,length(scale_beta));

for i = 1:length(scale_beta)
    tic;
    beta_spec.scale = scale_beta(i);
    beta_spec.shape = par_gam_beta(1)*par_gam_beta(2)/scale_beta(i);
    [ Opt_U_4(i),Opt_epsilon_max_4(i), Opt_num_4(i),U_4(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_spec,Alpha,Mu,Ref,Num,C,W_homo_spec,iteration,Epsilon_searchrange);
    toc
end
%}

    %{
lambda_scale = 0.4:0.1:0.8;
Opt_U_1 = zeros(1,length(lambda_scale));
Opt_epsilon_max_1 = zeros(1,length(lambda_scale));
Opt_num_1= zeros(1,length(lambda_scale));
U_1 = zeros(1,length(lambda_scale));


for i =1:length(lambda_scale)
    tic;
    
    lambda_spec.scale = lambda_scale(i);
    [ Opt_U_1(i),Opt_epsilon_max_1(i), Opt_num_1(i),U_1(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);

    toc;
end

plot(lambda_scale,Opt_epsilon_max_1);
%}

%{
lambda_shape = 3.5:0.5:5;
Opt_U_2 = zeros(1,length(lambda_shape));
Opt_epsilon_max_2 = zeros(1,length(lambda_shape));
Opt_num_2= zeros(1,length(lambda_shape));
U_2 = zeros(1,length(lambda_shape));

for i =1:length(lambda_shape)
    tic;
    
    lambda_spec.shape = lambda_shape(i);
    [ Opt_U_2(i),Opt_epsilon_max_2(i), Opt_num_2(i),U_2(i)] = ProspectTheoryBased_PTPara( lambda_spec,beta_homo_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);

    toc;
end

plot(lambda_shape,Opt_epsilon_max_2);
%}
%{
beta_scale = 0.04:0.01:0.07;
Opt_U_3 = zeros(1,length(beta_scale));
Opt_epsilon_max_3 = zeros(1,length(beta_scale));
Opt_num_3= zeros(1,length(beta_scale));
U_3 = zeros(1,length(beta_scale));


for i =1:length(beta_scale)
    tic;
    
    beta_spec.scale = beta_scale(i);
    [ Opt_U_3(i),Opt_epsilon_max_3(i), Opt_num_3(i),U_3(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);

    toc;
end

plot(beta_scale,Opt_epsilon_max_3);
%}

%{
beta_shape = 12:15;
Opt_U_4 = zeros(1,length(beta_shape));
Opt_epsilon_max_4 = zeros(1,length(beta_shape));
Opt_num_4= zeros(1,length(beta_shape));
U_4 = zeros(1,length(beta_shape));

for i =1:length(beta_shape)
    tic;
    
    beta_spec.shape = beta_shape(i);
    [ Opt_U_4(i),Opt_epsilon_max_4(i), Opt_num_4(i),U_4(i)] = ProspectTheoryBased_PTPara( lambda_homo_spec,beta_spec,Alpha,Mu,Ref,Num,C,W,iteration,Epsilon_searchrange);

    toc;
end

plot(beta_shape,Opt_epsilon_max_4);
%}

