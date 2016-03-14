
X = [-2.5:0.02:1];
Y = [-1:0.02:1];



max_iter=100;

M=[length(X),length(Y)];

for ii=1:length(X)    
    for jj=1:length(Y)
        con=0;
        Z=0;
      for zz = 1:100        
        C= X(ii)+Y(jj)*i;
        Z=Z^2+C
        M(ii,jj)=con;
        if( isinf(Z) || con==99 )
           % disp("Es infinito");
           % ii
           % jj
            break;
        end
        con=con+1;
      end
    end
end

