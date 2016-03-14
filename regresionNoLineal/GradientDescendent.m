function [theta,J_history] = GradientDescendent(X,Y,theta,m,alpha,iterations)

% Gradient Descendent Algorithm one characteristic
J_history = zeros(iterations, 1); % History plot (J vs iterations)

% Iterative

for i = 1 : iterations

    theta = theta - ( alpha / m ) * ( X' * ( X * theta - Y ) );

    % Save the cost J in every iteration    
    J_history(i) = CostFunction(X, Y, theta,m);
end


   

end



