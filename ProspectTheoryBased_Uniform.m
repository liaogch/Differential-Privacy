function [ Opt_U,Opt_epsilon_max, Opt_num] = ProspectTheoryBased_Uniform( Lambda,Beta,Alpha,Mu,Ref,Num,C,Wm,Wl,Epsilon_searchrange )

lambda = Lambda;
beta = Beta;
alpha = Alpha;
%From "advanced in prospect theory cumulative representation of uncertainty"

m = 1000;
u = Mu;
%{
N = Num;
%n_target = N/3*2; %data amount utility is 2/3 at this number
n_target = 600; %data amount utility is 2/3 at this number

k = 1.109;
%k = 0.273;
%g = 0.271;
l=(3*1.109-1)/n_target;
%}

N = Num;
%n_target = N/3*2; %data amount utility is 2/3 at this number
n_target = 600; %data amount utility is 2/3 at this number

%k = 0.909;
k = 0.989;
%k = 0.273;
%g = 0.271;
l=(3*1.109-1)/n_target;


c = C;

%theta = 0.05; 

%W_max = 1;

%{
if strcmp(opt, 'truncated')
    mu = 0.5;
    sigma = 1;
    nd=makedist('normal','mu',mu,'sigma',sigma);
    td=truncate(nd,0,inf);
    iter = 10;
    Wi = random(td,N*iter,1);
end
%}


%Wm = Wm;
%Wl = Wl;

W_min = Wm-Wl;
W_max = Wm+Wl;


epsilon_ref = Ref;
%{
M = c * lambda /i*power(1/i,beta);
temp = 0; 
for i = 1:i
    temp = temp + power(i,beta);
end
 
M = M*temp;
%}

%endpoint = W_max/M;

%epsilon_max = opt_eps_max1;
%epsilon_max = 0.002:0.0001:endpoint;
epsilon_max = Epsilon_searchrange;
prospect_val_parti = zeros(1,length(epsilon_max));
prospect_val_nonparti = zeros(1,length(epsilon_max));
sum = zeros(1,length(epsilon_max));
norm = zeros(1,length(epsilon_max));
G_parti = zeros(1,length(epsilon_max));
G_nonparti = zeros(1,length(epsilon_max));
G_dif = zeros(1,length(epsilon_max));
num = zeros(1,length(epsilon_max));
U_c = zeros(1,length(epsilon_max));
R_num = zeros(1,length(epsilon_max));
Acc = zeros(1,length(epsilon_max));
l_v = zeros(1,length(epsilon_max));
%g_num = zeros(1,length(epsilon_max));

Opt_U = -1000;
Opt_epsilon_max = 0;
Opt_index = 0;
Opt_num = 0;
Opt_G = 0;



%p = parpool(4);
%ttime = tic;

for i = 1:length(epsilon_max)
    
    sum(i) = 0;
    norm(i) = 0;
    for j = 1:m
        p = 1/m;
        epsilon = epsilon_max(i)/m*j;
        norm(i) = norm(i) + Weighting_Fun( p, u );
        sum(i) = sum(i) + Weighting_Fun( p, u )* Valuation_Fun( epsilon,beta,lambda,alpha,epsilon_ref);
    end

    prospect_val_parti(i) = sum(i) / norm(i);
    
    prospect_val_nonparti(i) = power(epsilon_ref,alpha);
    
    G_parti(i) = c * prospect_val_parti(i);
    
    G_nonparti(i) = c * prospect_val_nonparti(i);
    
    G_dif(i) = G_nonparti(i)-G_parti(i);
      
    %{
    if strcmp(opt, 'truncated')
        
        nn=0;    
        len = length(Wi);
        for ii=1:len
            if Wi(ii)>G_dif(i)
                nn = nn+1;
            end
        end
        num(i) = floor(nn/iter);
    else
    %}
        if G_dif(i)<W_min
            num(i)=N;
        else if G_dif(i)>=W_min && G_dif(i)<=W_max
                num(i) = N*(W_max-G_dif(i))/2/Wl;
            else
                num(i) = 0;
            end
        end
    
    
    
    
%{
    if G(m) < W_max
        %num(m) = floor(N * (1- G(m)/W_max)); 
        num(m) = N * (1- G(m)/W_max);
        
    else
        num(m) = 0;
        U_c(m) = 0;
        break;
    end
%}
    %k = 0.109;
    %R_num(m) = k*log(1+g*num(m));

    R_num(i) = 1 - k/(1+l*num(i));
    
    %g_num(m) = l * num(m);
    
    %R_num = 1 - u*exp(-h*num(m));
    if R_num(i) < 0
        R_num(i) = 0;
    end
    
    S_f = 1 / num(i);
    
    l_v(i) = S_f / epsilon_max(i);
    
    %gamma = -l_v(m) * log(theta);
    
    %gamma = l_v(m);
    
    gamma = 2*l_v(i)^2;
    
    Acc(i) = gamma;
    
    U_c(i) = R_num(i) - Acc(i);
   

end

%save 'data';

for i = 1:length(epsilon_max)
    if U_c(i) > Opt_U
        Opt_U = U_c(i);
        Opt_epsilon_max = epsilon_max(i);
        Opt_num = num(i);
        %Opt_index = m;
    end
end
%opt_eps_max = [opt_eps_max Opt_epsilon_max];

%opt_Uc = [opt_Uc Opt_U];
%delete(p);
%toc(ttime);

%hold on;
%figure;
%plot(epsilon_max,U_c,'-');

%{
beta1 = 1; 
M = c * lambda /n*power(1/n,beta1);
temp = 0; 
for i = 1:n
    temp = temp + power(i,beta1);
end
 
M = M*temp;

C = k*N/2/g/power(log(theta),2);
A = M/W_max;
i = 1;
epsilon_temp = 0:0.01:0.05;
deri = zeros(1,length(epsilon_temp));

for i = 1:length(epsilon_temp)
    deri(i) = 1 - A*C*beta1*epsilon_temp(i)^(beta1 + 2) - A*epsilon_temp(i)^beta1 + C*A*A*beta1*epsilon_temp(i)^(2*beta1 + 2) - beta1*A*epsilon_temp(i)^beta1;
end

figure(2)
plot(epsilon_temp,deri);
%}

%saveresult([ Opt_U,Opt_epsilon_max, Opt_num],'E:\liao\MATLAB\Differential Privacy','data','1','.txt');


end

