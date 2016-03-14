close all
x = load('./ex2Data/ex2x.dat');
y = load('./ex2Data/ex2y.dat');


%graficamos el DataSet
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')
hold on

n=3;%numero de caracteristicas.
m= length(y);%numero de muestras.


%1. Sobre el dataSet Grafique la funcion H(W) para valores 
%   de W aleatorios.
X=[ones(m,1) x];
W = [rand() ;rand()];
H=X*W;
plot(x,H,'g');
legend('Dataset', 'H(W), W aleatorios');


%2. Implemente el modelo H(x)=x^n;
phi=inline('x.^n','x','n');
Xn=[ones(m,1) phi(x,n)];
W = [rand() ;rand()]
%figure
%H=Xn*W;
%plot(x,H,'r');
%legend('H(W)=x^n, W aleatorios');

%3. Normalice los datos. 
xphi=phi(x,n);
S=std(xphi); %Desviacion estandar
U=mean(xphi); %Promedio/ media
xphi=(xphi-U)./S; %Normalizacion
    
X=[ones(m,1) xphi];



MAX_ITR = 4000;
alpha = 0.6;

W_plot=[];
J = [];
for num_iterations = 1:MAX_ITR
    grad = (1/m).* X' * ((X*W) - y);
    W = W - alpha .* grad;
    W_plot=[W_plot W];
    J = [J (0.5/m) .* ((X*W) - y)' * ((X*W) - y)];
end

%3. Grafique el modelo sobre el dataset con los nuevos parametros W
%   y la funcion de costo J.
figure
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')
hold on
plot(x,X*W);
legend('Dataset,Regresion_Normalizacion');

figure
plot([1:1:num_iterations],J, 'r')
legend('J');



%%
% Calculate J matrix

% Grid over which we will calculate J
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);
% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
	  for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = (0.5/m) .* ((X * t) - y)' * ((X * t) - y);
    end
end

% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

figure;
% Plot the cost function with 15 contours spaced logarithmically
% between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, linspace(-2, 2, 15))
xlabel('\theta_0'); ylabel('\theta_1')
hold on
plot(W_plot(1,:),W_plot(2,:),'*')