
function [J w_2 w_3] = backpropagation(X,Y,w_2,w_3,ITER,alpha,Nout) 

warning ("off", "Octave:broadcast");

m = size(X,1); % Muestras.
n = size(X,2); % Caracteristicas.

J = zeros(ITER,Nout); % vector que almacenara funcion de costo.
Je= zeros(ITER,Nout); % vector que almacenara el error por cada iteracion.

check = randi([1,ITER]); 

for i=1: ITER

        [h a_2 a_3] = forwardpropagation(X,w_2,w_3,m);

        h_prima = a_3 .* ( 1 - a_3 );
        a_prima = a_2 .* ( 1 - a_2 );
       
        s_1 = ( h - Y ) .* h_prima;
        s_0 = ( s_1 * w_3' ) .* a_prima;
       
        dw_3 = a_2' * s_1;
        dw_2 = X' * s_0(:,(2:size(s_0,2)));
        % Chequeo de gradiente para 1 iteracion:
        if(i == check)
            diff = gradientChecking(X,Y,w_2,w_3,dw_2,dw_3,alpha,Nout);        
        end;
        

        w_3 = w_3 - alpha .* dw_3;
        w_2 = w_2 - alpha .* dw_2;
            
        J(i,:) = costfunction(Y,h,m);
        
        hold on;
        if(i>1)
        plot([i-1,i],[J(i-1),J(i)],'-');
        drawnow
        else
        plot(i,J(i),'-');
        drawnow
        end;
        
        Je(i,:) = predict(h,Y,m);
        
end; 


figure 2;
plot([1:ITER],J);
legend('J_grad');
print('J_grad','-dpng');

figure 3;
plot([1:ITER],Je);
legend('E_grad');
print('E_grad','-dpng');


end;
