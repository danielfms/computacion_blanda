function [theta,J_history] = GradientDescendent(X,Y,theta,m,alpha,iterations)

% Gradient Descendent Algorithm one characteristic
J_history = zeros(iterations, 1); % History plot (J vs iterations)

% Iterative

for i = 1 : iterations
    sum0=0.0;
    sum1=0.0;
    for j = 1 : m    
	    h = theta(1) + (theta(2) * X(j));
	    sum0= sum0 + (h-Y(j));   
	    sum1= sum1+ ((h-Y(j)) * X(j));
    end
    
    
    % Simultaneous Update.
    theta_temp1 = theta(1) - ((alpha/m) *sum0);
    theta_temp2 = theta(2) - ((alpha/m) *sum1);
    theta(1) = theta_temp1;
    theta(2) = theta_temp2;
   
    % Save the cost J in every iteration    
    J_history(i) = CostFunction(X, Y, theta,m);
end


   

end



