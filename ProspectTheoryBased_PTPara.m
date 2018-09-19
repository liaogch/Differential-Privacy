function [ Opt_U,Opt_epsilon_max, Opt_num] = ProspectTheoryBased_PTPara( Lambda_spec,Beta_spec,Alpha,Mu,Ref,N,C,W_spec,iteration,Epsilon_searchrange )

%Opt_U  = zeros(1,iteration);
%Opt_epsilon_max = zeros(1,iteration);
%Opt_num = zeros(1,iteration);

%{
for jj = 1:iteration
    
    Lambda_spec.N = N;
    Beta_spec.N = N;
    W_spec.N = N;

    lambda_N = PT_Parameter_Generation( Lambda_spec);
    beta_N = PT_Parameter_Generation( Beta_spec);
    W_N = W_Parameter_Generation(W_spec);

    alpha = Alpha;
%}
%lambda = Lambda;
%beta = Beta;
%alpha = Alpha;

%From "advanced in prospect theory cumulative representation of uncertainty"
%m=100

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

%n_target = N/3*2; %data amount utility is 2/3 at this number

    n_target = 600; %data amount utility is 2/3 at this number

%k = 0.909;
%k = 0.273;
%g = 0.271;
    k = 0.989;

    l=(3*1.109-1)/n_target;



    c = C;

%theta = 0.05; 

%W_max = 1;



%{
    nd=makedist('normal','mu',average,'sigma',sigma);
    td=truncate(nd,lower,upper);
    Wi = random(td,N*iteration,1);
%}



%Wm = Wm;
%Wl = Wl;
%{
W_min = Wm-Wl;
W_max = Wm+Wl;
%}

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
%prospect_val_parti = zeros(1,length(epsilon_max));
%prospect_val_nonparti = zeros(1,length(epsilon_max));
%sum = zeros(1,length(epsilon_max));
%norm = zeros(1,length(epsilon_max));
%G_parti = zeros(1,length(epsilon_max));
%G_nonparti = zeros(1,length(epsilon_max));
%G_dif = zeros(1,length(epsilon_max));
    num = zeros(1,length(epsilon_max));
    U_c = zeros(1,length(epsilon_max));
    R_num = zeros(1,length(epsilon_max));
    Acc = zeros(1,length(epsilon_max));
    l_v = zeros(1,length(epsilon_max));
%g_num = zeros(1,length(epsilon_max));

%{
Opt_U = -1000;
Opt_epsilon_max = 0;
Opt_index = 0;
Opt_num = 0;
Opt_G = 0;
%}


%p = parpool(4);
%ttime = tic;

for i = 1:length(epsilon_max)
    %{
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
      %}
    

        %counting number of participants
        
        %len = N;
        nn=0;    
        for jj = 1:iteration
    
            Lambda_spec.N = N;
            Beta_spec.N = N;
            W_spec.N = N;

            lambda_N = PT_Parameter_Generation( Lambda_spec);
            beta_N = PT_Parameter_Generation( Beta_spec);
            W_N = W_Parameter_Generation(W_spec);

            alpha = Alpha;
            
            
            for ii=1:N
            %{
            if check_participation( lambda_N(ii),beta_N(ii),alpha,epsilon_ref,W_N(ii),c,epsilon_max,m)==1
                nn = nn+1;
             end
            %}
                m = 100;
                sum = 0;
            %norm = 0;
                for j = 1:m
                %p = 1/m;
                    epsilon = epsilon_max(i)/m*j;
                %norm = norm + Weighting_Fun( p, u );
                %sum(i) = sum(i) + Weighting_Fun( p, u )* Valuation_Fun( epsilon,beta,lambda,alpha,epsilon_ref);
                    sum = sum + Valuation_Fun( epsilon,beta_N(ii),lambda_N(ii),alpha,epsilon_ref);
                end

            prospect_val_parti = sum / m;
    
            prospect_val_nonparti = power(epsilon_ref,alpha);
    
            G_parti = c * prospect_val_parti;
    
            G_nonparti = c * prospect_val_nonparti;
    
            G_dif = G_nonparti-G_parti;
    
                if W_N(ii)>G_dif
                    nn = nn+1;
                end
           
        
            end
        end

        
        num(i) = floor(nn/iteration);
    %{
        if G_dif(i)<W_min
            num(i)=N;
        else if G_dif(i)>=W_min && G_dif(i)<=W_max
                num(i) = N*(W_max-G_dif(i))/2/Wl;
            else
                num(i) = 0;
            end
        end
    %}
    
    
    
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
    
    r = 1;
    
    gamma = r*2*l_v(i)^2;
    
    Acc(i) = gamma;
    
    U_c(i) = R_num(i) - Acc(i);

    
end

    [Opt_U, Opt_index] = max(U_c);
    Opt_epsilon_max = epsilon_max(Opt_index);
    Opt_num = num(Opt_index);
%{
    [Opt_U(jj), Opt_index] = max(U_c);
    Opt_epsilon_max(jj) = epsilon_max(Opt_index);
    Opt_num(jj) = num(Opt_index);
%}
%end

%Opt_U_ave = mean(Opt_U);
%Opt_epsilon_max_ave = mean(Opt_epsilon_max);
%Opt_num_ave = floor(mean(Opt_num));

%save 'data';
%{
for i = 1:length(epsilon_max)
    if U_c(i) > Opt_U
        Opt_U = U_c(i);
        Opt_epsilon_max =  epsilon_max(i);
        Opt_num = num(i);
        %Opt_index = m;
    end
end
%}
%opt_eps_max = [opt_eps_max Opt_epsilon_max];


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





