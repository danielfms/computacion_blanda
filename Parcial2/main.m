clear all;
format long;
t0=clock(); % Funcion de tiempo

data=load("./in/juan.mat");


number_data = 100;

X=data.X([1:number_data],:);
Y=data.t(1:number_data);
plot(X(:,1),X(:,2),'o');
%:::::::::::::::::::::::::::::::::::::::::::::::::::2. Forward propagation


m = size(X,1); % Muestras;


y=zeros(m,2);
for i=1:m
         if(Y(i)==-1)
            y(i,Y(i)+2)=1; % 1 , 0
         else
             y(i,Y(i)+1)=1; % 0 , 1
         end ;
end;

h= zeros(m,1); % Modelo de salida; 
n = size(X,2);  % Caracteristicas;
N_0 = 2; % Neuronas Capa entrada;
N_1 = 10;  % Neuronas Capa ocualta;
N_2 = 2;   % Neuronas Capa salida;
ITER = 200;
alpha= 0.0001;

epsilon=1;
%w_0 = ones(N_1+1,N_2);
%w_1 = ones(n+1,N_1);
w_0 = rand(n+1,N_1)*(2*epsilon) - epsilon;
%a = load('w0rand_grad');
w_1 = rand(N_1+1,N_2)*(2*epsilon) - epsilon;
%b = load('w1rand_grad');
%w_0 = a.w_0;
%w_1 = b.w_1;
%w_1 =[ones(N_1+1,1) w_1];
X=[ones(m,1) X];


%save w_0rand w_0;
%save w_1rand w_1;

% GRADIENTCHECKING backpropagation para sacar derivadas para hacer gradient checking.
[J w_0 w_1 dw_0 dw_1] = backpropagation(X,y,w_0,w_1,ITER,alpha,N_2);
%disp("Derivadas calculadas!");
[h a_0 a_1] = forwardpropagation(X,w_0,w_1);

% ::::::::::::::::::::::::::::::::::::::3. Backpropagation 

[v hclass] = max(h,[],2);

for i=1:m
    if(hclass(i)== 2)
        hclass(i)=1;
    else
        hclass(i)=-1;
    end;
end

correctas=0;



for i=1:m
    if(hclass(i)== Y(i))
        correctas++;
    end;    
end

disp("Correctas:");
disp(correctas);
elapsed_time = etime (clock (), t0);
disp("Tiempo :");
disp(elapsed_time);
disp("Iteraciones :");
disp(ITER);
disp("Alpha :");
disp(alpha);
