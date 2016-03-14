function err = predict(h,Y,m) 

for i=1:m
    if(h(i) >= 0.5)
        h(i)=1;
    else
        h(i)=0;
    end;    
end


correctas=0;

for i=1:m
    if(h(i) == Y(i))
        correctas++;
    end;    
end;
err = (m-correctas)/m;

end;