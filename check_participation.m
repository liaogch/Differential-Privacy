function [ res ] = check_participation( lambda,beta,alpha,epsilon_ref,W,c,epsilon_max,m)
    sum = 0;
    for j = 1:m
                
        epsilon = epsilon_max/m*j;
         
        sum = sum + Valuation_Fun( epsilon,beta,lambda,alpha,epsilon_ref);
    end

    prospect_val_parti = sum / m;
    
    prospect_val_nonparti = power(epsilon_ref,alpha);
    
    if W> c*prospect_val_nonparti-c*prospect_val_parti;
       res = 1;
    else
       res = 0;
    end
end

