function err = predict(h,Y,m) 

[v hclass] = max(h,[],2);
[v2 yclass] = max(Y,[],2);


%for i=1:m
%    if(hclass(i)== 2)
%        hclass(i)=1;
%    else
%        hclass(i)=-1;
%    end;
%end

correctas=0;

for i=1:m
    if(hclass(i)== yclass(i))
        correctas++;
    end;    
end;
err = (m-correctas)/m;

end;