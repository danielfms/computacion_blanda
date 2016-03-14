
function [J w_0 w_1 dw_0 dw_1] = backpropagation(X,Y,w_0,w_1,ITER,alpha,nout) 

warning ("off", "Octave:broadcast");

m = size(X,1); %Muestras
n = size(X,2);  % Caracteristicas

J= zeros(ITER,nout);
Je= zeros(ITER,nout);


for i=1: ITER

        [h a_0 a_1] = forwardpropagation(X,w_0,w_1);

        h_prima = a_1 .* ( 1 - a_1 );
        a_prima = a_0 .* ( 1 - a_0 );
       
        s_1 = ( a_1 - Y ) .* h_prima;
        s_0 = ( s_1 * w_1' ) .* a_prima;
       
        dw_1 = a_0' * s_1;
        final=size(s_0,2);
        dw_0 = X' * s_0(:,(2:final));
        
        % Gradient check 
        %diff = gradientChecking(X,Y,w_0,w_1,dw_0,dw_1,alpha);        

        w_1 = w_1 - alpha .* dw_1;
        w_0 = w_0 - alpha .* dw_0;
            
        J(i,:)= costfunction(Y,h,m);
        
        Je(i,:)=predict(h,Y,m);
        
end; 

%disp("Verificación del gradiente: ");
%disp(diff);

figure 1;
plot([1:ITER],J);
legend('J_grad');
print('J_grad','-dpng');


figure 2;
plot([1:ITER],Je);
legend('E_grad');
print('E_grad','-dpng');


end;
