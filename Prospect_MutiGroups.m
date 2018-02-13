

Group_num = 2;

lambda_1 = 2.25;
beta_1 = 0.88;
alpha_1 = 0.88;
%From "advanced in prospect theory cumulative representation of
%uncertainty"sfsddfsdfsdf 

N_1 = 1000;
u_1 = 0.65;
epsilon_ref_1 = 0 ;

lambda_2 = 3;
beta_2 = 0.88;
alpha_2 = 0.88;
N_2 = 1000;
u_2 = 0.65;
epsilon_ref_2 = 0 ;

m = 1000;
%N = 1000;
%n_target = N/3*2; %data amount utility is 2/3 at this number
n_target = 600; %data amount utility is 2/3 at this number

%k=1.5;
k = 1.109;
%k = 0.273;
%g = 0.271;
l=(3*1.109-1)/n_target;


c = 1;

%theta = 0.05; 

%W_max = 1;

Wm = 0.5;
Wl = 0.5;

W_min = Wm-Wl;
W_max = Wm+Wl;



data_max = 1;%maximum value of the collected data

le = 1; %Accuracy loss(e) = le * e^2;


%{
M = c * lambda /n*power(1/n,beta);
temp = 0; 
for i = 1:n
    temp = temp + power(i,beta);
end
 
M = M*temp;


endpoint = W_max/M;
%}

epsilon_max = 0.002:0.0001:0.09;
%epsilon_max = 0.0183;

prospect_val_1 = zeros(1,length(epsilon_max));
sum_1 = zeros(1,length(epsilon_max));
norm_1 = zeros(1,length(epsilon_max));
G_1 = zeros(1,length(epsilon_max));
num_1 = zeros(1,length(epsilon_max));
prospect_val_parti_1 = zeros(1,length(epsilon_max));
prospect_val_nonparti_1 = zeros(1,length(epsilon_max));
G_parti_1 = zeros(1,length(epsilon_max));
G_nonparti_1 = zeros(1,length(epsilon_max));
G_dif_1 = zeros(1,length(epsilon_max));

prospect_val_2 = zeros(1,length(epsilon_max));
sum_2 = zeros(1,length(epsilon_max));
norm_2 = zeros(1,length(epsilon_max));
G_2 = zeros(1,length(epsilon_max));
num_2 = zeros(1,length(epsilon_max));
prospect_val_parti_2 = zeros(1,length(epsilon_max));
prospect_val_nonparti_2 = zeros(1,length(epsilon_max));
G_parti_2 = zeros(1,length(epsilon_max));
G_nonparti_2 = zeros(1,length(epsilon_max));
G_dif_2 = zeros(1,length(epsilon_max));


U_c = zeros(1,length(epsilon_max));
R_num = zeros(1,length(epsilon_max));
Acc_loss = zeros(1,length(epsilon_max));
b = zeros(1,length(epsilon_max));
g_num = zeros(1,length(epsilon_max));
num = zeros(1,length(epsilon_max));


Opt_U = -1000;
Opt_epsilon_max = 0;
Opt_index = 0;
Opt_num = 0;
Opt_G = 0;
Opt_Acc = 0;


for i = 1:length(epsilon_max)
    
    sum_1(i) = 0;
    norm_1(i) = 0;
    for j = 1:m
        p = 1/m;
        epsilon = epsilon_max(i)/m*j;
        norm_1(i) = norm_1(i) + Weighting_Fun( p, u_1 );
        sum_1(i) = sum_1(i) + Weighting_Fun( p, u_1 )* Valuation_Fun( epsilon,beta_1,lambda_1,alpha_1,epsilon_ref_1);
    end

    prospect_val_parti_1(i) = sum_1(i) / norm_1(i);
    
    prospect_val_nonparti_1(i) = power(epsilon_ref_1,alpha_1);
    
    G_parti_1(i) = c * prospect_val_parti_1(i);
    
    G_nonparti_1(i) = c * prospect_val_nonparti_1(i);
    
    G_dif_1(i) = G_nonparti_1(i)-G_parti_1(i);
      
    
    if G_dif_1(i)<W_min
        num_1(i)=N;
    else if G_dif_1(i)>=W_min && G_dif_1(i)<=W_max
            num_1(i) = floor(N_1*(W_max-G_dif_1(i))/2/Wl);
        else
            num_1(i) = 0;
        end
    end
    %Calculation for group 1
    
    sum_2(i) = 0;
    norm_2(i) = 0;
    for j = 1:m
        p = 1/m;
        epsilon = epsilon_max(i)/m*j;
        norm_2(i) = norm_2(i) + Weighting_Fun( p, u_2 );
        sum_2(i) = sum_2(i) + Weighting_Fun( p, u_2 )* Valuation_Fun( epsilon,beta_2,lambda_2,alpha_2,epsilon_ref_2);
    end

    prospect_val_parti_2(i) = sum_2(i) / norm_2(i);
    
    prospect_val_nonparti_2(i) = power(epsilon_ref_2,alpha_2);
    
    G_parti_2(i) = c * prospect_val_parti_2(i);
    
    G_nonparti_2(i) = c * prospect_val_nonparti_2(i);
    
    G_dif_2(i) = G_nonparti_2(i)-G_parti_2(i);
      
    
    if G_dif_2(i)<W_min
        num_2(i)=N_2;
    else if G_dif_2(i)>=W_min && G_dif_2(i)<=W_max
            num_2(i) = floor(N_2*(W_max-G_dif_2(i))/2/Wl);
        else
            num_2(i) = 0;
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
    
    num(i) = num_1(i)+num_2(i);

    R_num(i) = 1 - k/(1+l*num(i));
    
    g_num(i) = l * num_1(i);
    
    %R_num = 1 - u*exp(-h*num(m));
    if R_num(i) < 0
        R_num(i) = 0;
    end
    
    S_f = data_max / num_1(i);
    
    b(i) = S_f / epsilon_max(i);
    
    %gamma = -l_v(m) * log(theta);
    
    %gamma = l_v(m);
    
    %gamma = 2*b(m)^2;
    
    %Acc_loss(m) = gamma;
    
    Acc_loss(i) = le*2*b(i)^2;
    
    U_c(i) = R_num(i) - Acc_loss(i);

    if U_c(i) > Opt_U
        Opt_U = U_c(i);
        Opt_epsilon_max = epsilon_max(i);
        Opt_num = num_1(i);
        Opt_index = i;
        Opt_G = G_1(i);
        Opt_Acc = Acc_loss(i);
    end


end
%opt_eps_max = [opt_eps_max Opt_epsilon_max];

%opt_Uc = [opt_Uc Opt_U];

hold on;

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
