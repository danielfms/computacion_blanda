function diff = gradientChecking(X,Y,w_0,w_1,dw_0,dw_1,alpha,Nout) 
    
    m = size(X,1); % Muestras
    theta = [w_0(:);w_1(:)];    
    grad = [dw_0(:);dw_1(:)];
    size(grad)
    l = size(theta,1);
    gradnum = zeros(l,1);
    eps = 1e-4;
    

    rows0 = size(w_0,1);
    cols0 = size(w_0,2);
    rows1 = size(w_1,1);
    cols1 = size(w_1,2);
    
    
    for i=1:l
        
        w_plus = theta;
        w_plus(i) = w_plus(i) + eps;
        w_minus= theta;
        w_minus(i)= w_minus(i) - eps;

        %Forward propagation w_plus      
        w_0c=reshape(w_plus(1: rows0*cols0),rows0,cols0);
        w_1c=reshape(w_plus(rows0*cols0+1: rows0*cols0+rows1*cols1),rows1,cols1);
        
        [ghc a_0 a_1]= forwardpropagation(X,w_0c,w_1c,m);
        grad_plus = costfunction(Y,ghc,m);

        %Forward propagation w_minus
        w_0c=reshape(w_minus(1: rows0*cols0),rows0,cols0);
        w_1c=reshape(w_minus(rows0*cols0+1: rows0*cols0+rows1*cols1),rows1,cols1);
        
        [ghc a_0 a_1]= forwardpropagation(X,w_0c,w_1c,m);
        grad_minus = costfunction(Y,ghc,m);
        %size(grad_plus)
        %Construccion gradnum
        %size( grad_plus - grad_minus) / ( 2 * eps )
        gradnum(i) = ( grad_plus - grad_minus) / ( 2 * eps );

    end;
   
   % Gradient Checking
    diff = norm(grad-gradnum)/norm(grad+gradnum);
    disp("Gradient Checking: ");
    disp(diff); 

