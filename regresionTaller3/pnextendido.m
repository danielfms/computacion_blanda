close all
x = load('./ex2Data/ex2x.dat');
y = load('./ex2Data/ex2y.dat');


%graficamos el DataSet
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')
hold on

n=10;%numero de caracteristicas.
m= length(y);%numero de muestras.


%1. Sobre el dataSet Grafique la funcion H(W) para valores 
%   de W aleatorios.
X=[ones(m,1) x];
W = [rand() ;rand()];
H=X*W;
plot(x,H,'g');
legend('Dataset', 'H(W), W aleatorios');


%2. Implemente el modelo H(x)=w0x^0 + w1x^1 + w2x^1 + ... wn^n;
phi=inline('x.^n','x','n');
Xn=[ones(m,1)];
W= [rand()];
% Creo matriz X
for i=1:n
    Xn= [Xn phi(x,i)];
    W= [W rand()];
end
W=W';

%3. Normalice los datos.
for i=2:n+1
    xphi=Xn(:,i);
    S=std(xphi); %Desviacion estandar
    U=mean(xphi); %Promedio/ media
    Xn(:,i)=(xphi-U)/S; %Normalizacion 
end

X=Xn;

MAX_ITR = 4000;
alpha = 0.06;

W_plot=[];
J = [];
for num_iterations = 1:MAX_ITR
    grad = (1/m).* X' * ((X*W) - y);
    W = W - alpha .* grad;
    W_plot=[W_plot W];
    J = [J (0.5/m) .* ((X*W) - y)' * ((X*W) - y)];
end

%4. Grafique el modelo sobre el dataset con los nuevos parametros W
%   y la funcion de costo J.
figure
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')
hold on
plot(x,X*W,'r');
legend('Dataset','Regresion _ Normalizacion');

figure
plot([1:1:num_iterations],J, 'r')
legend('J');
