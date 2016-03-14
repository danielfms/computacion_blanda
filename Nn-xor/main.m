%::::::::::::::::::::::::::::::::
% Implementaci'on red neuronal para 
% clasificar superficies no lineales.
% Problema XOR.
% Dataset tomado de https://cw.fel.cvut.cz/wiki/courses/y33aui/cviceni/cviceni05
%::::::::::::::::::::::::::::::::

% ::::::::::::::::::::::::::::::::::::::::::::::::: 1. Inicializacion Workspace

clear all, close all;

format long; % Mayor precision.
t0 = clock(); % Funcion de tiempo.

% ::::::::::::::::::::::::::::::::::::::::::::::::::: 1.1 Carga de datos

data = load('./in/dataxor.txt');
X = (data(:,[1:2]));
Y = (data(:,3));
clear data; % Liberamos memoria

% :::::::::::::::::::::::::::::::::::::::::::::::::::1.2 Dibujamos dataset

figure 1
title ('X');
plot(X,'o');
figure 2
title ('X vs Y');
plot(X(1:10),Y(1:10),'o');
hold on;
plot(X(3000:3010),Y(3000:3010),'o');

% :::::::::::::::::::::::::::::::::::::::::::::::::: 2. Forward propagation

% :::::::::::::::::: 2.1 Caracteristica X0

X=[ones(size(X,1),1) X];

% :::::::::::::::::: 2.2 Datos de la red

m = size(X,1); % Tamano de las muestras (+1 por la caracteristica X0);
n = size(X,2);  % Caracteristicas;
N_2 = 2;  % Neuronas capa oculta;
N_3 = 1;   % Neuronas capa salida;
h = zeros(m,N_3); % Modelo de salida; 
ITER = 1000;  % Iteraciones para refinamiento;
alpha= 0.001; % Factor de aprendizaje.

% ::::::::::::::::::: 2.3 Generacion de pesos.

% Generacion de esta manera para eliminar problemas de simetria ("symmetry break").

epsilon = 1;
%w_2 = rand( n , N_2 ) * ( 2 * epsilon ) - epsilon; % descomentar para generar nuevos pesos
%w_3 = rand( N_2+1 , N_3 ) * ( 2 * epsilon ) - epsilon; descomentar para generar nuevos pesos

a = load('w_2'); % Comentar para generar nuevos pesos 
b = load('w_3'); % Comentar para generar nuevos pesos

w_2=a.w_2; % Comentar para generar nuevos pesos
w_3=b.w_3; % Comentar para generar nuevos pesos

% ::::::::::::::::: 2.4  Guardamos pesos optimos

save w_2 w_2;
save w_3 w_3;

% :::::::::::::::::::::::::::::::::::::::::::::::::::::: 3.  Back propagation

[J w_2 w_3] = backpropagation(X,Y,w_2,w_3,ITER,alpha,N_2);
[h a_2 a_3] = forwardpropagation(X,w_2,w_3,m);

% :::::::::::::::::::::::::::::::::::::::::::::::::::::: 4. Obtencion de aciertos 

% ::::::::::: 4.1 Debido a la naturaleza del dataset ( entre 0 y 1) predecimos valores

for i=1:m
    if(h(i) >= 0.5)
        h(i)=1;
    else
        h(i)=0;
    end;    
end

% ::::::::: 4.2  aciertos

correctas=0;
for i=1:m
    if(h(i)== Y(i))
        correctas++;
    end;    
end

E = (m-correctas)/m;

% ::::::::::::::::::::::::::::::::::::::::::::::::::::: 5. Resultados finales

elapsed_time = etime (clock (), t0); % Tiempo tomado por la implementacion.

disp("::::::::::::::::::: Datos de entrada ::::::::::::::::::::");
disp("Iteraciones :");
disp(ITER);
disp("Alpha :");
disp(alpha);

disp("::::::::::::::::::: Resultados ::::::::::::::::::::");
disp("Correctas:");
disp(correctas);
disp("Error Final:");
disp(E);
disp("Tiempo Tomado :");
disp(elapsed_time);