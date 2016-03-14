clear all;
format long;
t0=clock(); % Funcion de tiempo

%training data
data1=load('./in/data_batch_1.mat');
data2=load('./in/data_batch_2.mat');
data3=load('./in/data_batch_3.mat');  
data4=load('./in/data_batch_4.mat');
data5=load('./in/data_batch_5.mat');

dataX=[data1.data;data2.data;data3.data;data4.data;data5.data];
dataY=[data1.labels;data2.labels;data3.labels;data4.labels;data5.labels];
clear data1;clear data2;clear data3;clear data4; clear data5;

number_data = 10000;

%indices = randperm(number_data);
%datat=load('./in/test_batch.mat');
%Xtesting= double(datat.data(indices,:));
%Ytesting= double(datat.labels(indices));

%X = Xtesting;
%Y = Ytesting;

%X=double(dataX(indices,:));
%Y=double(dataY(indices));

X=double(dataX([1:number_data],:));
Y=double(dataY([1:number_data]));

clear dataX;dataY;





%plot(X(:,1),X(:,2),'o');
%:::::::::::::::::::::::::::::::::::::::::::::::::::2. Forward propagation


m = size(X,1); % Muestras;


y=zeros(m,10);
for i=1:m
         y(i,Y(i)+1)=1;
end;



h= zeros(m,10); % Modelo de salida; 
n = size(X,2);  % Caracteristicas;
N_0 =3072; % Neuronas Capa entrada;
N_1 = 300;  % Neuronas Capa ocualta;
N_2 = 10;   % Neuronas Capa salida;
ITER = 18000;
alpha= 0.00001;

epsilon=1;
%w_0 = ones(N_1+1,N_2);
%w_1 = ones(n+1,N_1);
%w_0 = rand(n+1,N_1)*(2*epsilon) - epsilon;
a = load('w0E38');
%w_1 = rand(N_1+1,N_2)*(2*epsilon) - epsilon;
b = load('w1E38');
w_0 = a.w_0;
w_1 = b.w_1;
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
[v yclass] = max(y,[],2);

correctas=0;
for i=1:m
    if(hclass(i)== yclass(i))
        correctas++;
    end;    
end

E = (number_data-correctas)/number_data;

disp("Correctas:");
disp(correctas);

disp("Error:");
disp(E);

elapsed_time = etime (clock (), t0);
disp("Tiempo :");
disp(elapsed_time);
disp("Iteraciones :");
disp(ITER);
disp("Alpha :");
disp(alpha);
