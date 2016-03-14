function Contour_3D(X,Y,m);
    fprintf('Plotting 3D. Press enter to continue.\n');
    pause;

    theta0=linspace(-3,3,100);
    theta1=linspace(-1,1,100);
    J_vals=zeros( length(theta0),length(theta1) );
    
    for i=1:length(theta0)
       for j=1:length(theta1)
            t=[theta0(i) ; theta1(j)];
            J_vals(i,j)=(0.5/m).*( ( (X*t)-Y )' * ( (X*t)-Y ) );    
        end
    end
    
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0, theta1, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

fprintf('Plotting Contour. Press enter to continue.\n');
pause;

% Plot the cost function with 15 contours spaced logarithmically
% between 0.01 and 100
contour(theta0, theta1, J_vals, linspace(-2, 2, 15))
xlabel('\theta_0'); ylabel('\theta_1')

