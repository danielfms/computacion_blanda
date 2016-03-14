function J = CostFunction(X,Y,theta,m)

% Algorithm CostFunction, One characteristic Iterative

acc = 0.000; % Accumulator

for i = 1:m
    h = theta(1) + ( theta(2) * X(i) ); 
    acc = acc + (( h - Y(i) )^2); 
end

% Calculated J  

J = (1 / (2*m) * acc);

end
