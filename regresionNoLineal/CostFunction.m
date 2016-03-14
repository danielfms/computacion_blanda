function J = CostFunction(X,Y,theta,m)

% Vectorize cost function

J=(1 / ( 2 * m ) ) .* ( X * theta - Y )' * ( X * theta - Y );

end
