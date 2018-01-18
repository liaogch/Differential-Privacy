lambda = 2.25;
beta = 1;
alpha = 1;
%From "advanced in prospect theory cumulative representation of uncertainty"

n = 1000;
u = 0.65;

N = 5000;
%n_target = N/3*2; %data amount utility is 2/3 at this number
n_target = 600; %data amount utility is 2/3 at this number

k = 0.989;
%k = 0.273;
%g = 0.271;
l=(3*1.109-1)/n_target;


c = 1;

theta = 0.05; 


Wm = 0.5;
Wl = 0.5;

W_max = Wm+Wl;
%{
mu = 1;
sigma = 1;

W_max = 2;
W_min = 0;
%}

epsilon_ref = 0;

M = c * lambda /n*power(1/n,beta);
temp = 0; 
for i = 1:n
    temp = temp + power(i,beta);
end
 
M = M*temp;


endpoint = W_max/M;

%epsilon_max = opt_eps_max1;
epsilon_max = 0.002:0.0001:0.03;
%epsilon_max = 0.0143;
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
g_num = zeros(1,length(epsilon_max));

Opt_U = -1000;
Opt_epsilon_max = 0;
Opt_index = 0;
Opt_num = 0;
Opt_G = 0;

ttime = tic;

%p = parpool(4);
for m = 1:length(epsilon_max)
    
    sum(m) = 0;
    norm(m) = 0;
    for i = 1:n
        p = 1/n;
        epsilon = epsilon_max(m)/n*i;
        norm(m) = norm(m) + Weighting_Fun( p, u );
        sum(m) = sum(m) + Weighting_Fun( p, u )* Valuation_Fun( epsilon,beta,lambda,alpha,epsilon_ref);
    end

    prospect_val_parti(m) = sum(m) / norm(m);
    
    prospect_val_nonparti(m) = power(epsilon_ref,alpha);
    
    G_parti(m) = c * prospect_val_parti(m);
    
    G_nonparti(m) = c * prospect_val_nonparti(m);
    
    G_dif(m) = G_nonparti(m)-G_parti(m);
    
    W_min = Wm-Wl;
    W_max = Wm+Wl;
    
    %{
    nd=makedist('normal','mu',mu,'sigma',sigma);
    td=truncate(nd,W_min,W_max);
    
    nn=0;    
    iter = 1;
    for ii=1:N*iter
        Wi = random(td,1,1);
        if Wi>G_dif(m)
            nn = nn+1;
        end
    end
    num(m) = floor(nn/iter);
    %}
    
    
    if G_dif(m)<W_min
        num(m)=N;
    else if G_dif(m)>=W_min && G_dif(m)<=W_max
            num(m) = N*(W_max-G_dif(m))/2/Wl;
        else
            num(m) = 0;
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

    R_num(m) = 1 - k/(1+l*num(m));
    
    g_num(m) = l * num(m);
    
    %R_num = 1 - u*exp(-h*num(m));
    if R_num(m) < 0
        R_num(m) = 0;
    end
    
    S_f = 1 / num(m);
    
    l_v(m) = S_f / epsilon_max(m);
    
    %gamma = -l_v(m) * log(theta);
    
    %gamma = l_v(m);
    
    gamma = 2*l_v(m)^2;
    
    Acc(m) = gamma;
    
    U_c(m) = R_num(m) - Acc(m);

    


end

for m = 1:length(epsilon_max)
    if U_c(m) > Opt_U
        Opt_U = U_c(m);
        Opt_epsilon_max = epsilon_max(m);
        Opt_num = num(m);
        Opt_index = m;
    end
end
opt_eps_max = [opt_eps_max Opt_epsilon_max];

%opt_Uc = [opt_Uc Opt_U];

%delete(p);
%hold on;
toc(ttime);

plot(epsilon_max,U_c,'-');

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
