 function state =saveresult(data,path,name,suffix)
%save simulation result data into a file defined by path/name_info.suffix
 %if SaveResult succeeds, state=1, else state=0
 %2014/08/20 17:15
n=[path name];
fullname=[n suffix];
if(exist(fullname)==0)    
    dlmwrite(fullname,data,'delimiter','\t','newline','pc');
    state=1;
else
    for i=1:100
         fullname=[n num2str(i) suffix];
        if(exist(fullname)==0)           
            dlmwrite(fullname,data,'delimiter','\t','newline','pc');
            state=1;
            return;
        end
    end
    state=0;
end
end
