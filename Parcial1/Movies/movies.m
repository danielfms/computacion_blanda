close all
tic;
%Como crear archivos binarios
%M=load('foo.csv');
%save -mat7-binary datosX M
%Y=load('Y');
%save -mat7-binary datosY Y


datosX = load('datosX');
Xs  = datosX.M;
datosY = load('datosY');
Ys= datosY.Y;

clear datosX; %Libero memoria
clear datosY; %Libero memoria

% Hacer funciones de calculo 80% y 20%
%Indices Aleatorios 80% training 20% testing
muestras= length(Xs);
indices = randperm(muestras);
muestras80= round(muestras*0.8);
muestras20= round(muestras*0.2);
indicestraining= indices(1:muestras80);    % 20000 datos de entrenamiento
indicestesting= indices(muestras80+1:muestras); % 5000 datos de prueba

X= Xs(indicestraining,:); 	      % Datos para entrenamiento
Y= Ys(indicestraining); 	      % Datos para entrenamiento
Xtesting= Xs(indicestesting,:);       % Datos para pruebas
Ytesting= Ys(indicestesting);       % Datos para pruebas
Xtesting = [ones(muestras20, 1) Xtesting];

%Normalizacion de los datos
features = columns(X);
S = max(X)-min(X);
U = mean(X); %promedio

for i=1:features
    X(:,i) = (X( : , i ) - U(i)) / S(i);
end

%graficamos el DataSet
%plot([1:25000], Y, 'o');
% Find Indices of Positive and Negative Examples
%pos = find(Y==1); 
%neg = find(Y == 0);
% Plot Examples
%plot(pos, Y(pos), 'k+','LineWidth', 2,'MarkerSize', 5);
%hold on
%plot(neg, Y(neg), 'go', 'MarkerFaceColor', 'y','MarkerSize', 5);
%ylabel('Reviews')
%xlabel('Sentiment')
%hold on

[m, n] = size(X);
#Add ones col
X = [ones(m, 1) X];
W=zeros(n+1,1);

MAX_ITR =1500;
alpha = 0.1;

J = 0;
J_history = zeros(MAX_ITR, 1);
E = [];

for i = 1:MAX_ITR
    Z=X*W;
    H= 1 ./ ( 1 + exp( -Z ));
    W = W - ( alpha / m )*( X' * ( H - Y ));
    J = ( - 1 / m ) * ( ( Y' * log(H+0.0000000001) ) + ( ( 1 - Y )' * log( 1 - H ) ) );
	J_history(i) = J;

    %Error
    Htesting= 1 ./ ( 1 + exp( -1 .* (Xtesting*W) ));
    Ycontrol= zeros(muestras20,1);
    cont=0;
    for i=1: muestras20
        if Htesting(i)>=0.5
            Ycontrol(i)=1;
        else
            Ycontrol(i)=0;
        end
    
        if Ycontrol(i) ==Ytesting(i)
            cont=cont+1;
        end;
    end

    E = [E (muestras20-cont)/muestras20];
 
end

%3. Grafique el modelo sobre el dataset con los nuevos parametros W
%   y la funcion de costo J.

%figure
plot(J_history,'r')
legend('J');
print('J','-dpng');

figure
plot(E,'r')
legend('E');
print('E','-dpng');

toc;
