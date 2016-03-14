% Initialization
clear ; close all; clc

% :::::::::::::: Load data set
X = load ("ex2x.dat");
Y = load ("ex2y.dat");
% :::::::::::::: Graph Data
fprintf('Plotting Data. Press enter to continue.\n');
pause;
plot(X,Y,'o');
title(' X vs Y ');
xlabel('X');
ylabel('Y');
fprintf('Program paused. Press enter to continue.\n');
pause;

% ::::::::::::: Gradient Descendent Iterative

m = length(Y);
theta = [0 0]; % initial values
iterations = 500;
alpha = 0.07;
% Compute and display initial cost
disp('Cost Function in theta = 0X+0X1 = '), disp(CostFunction(X,Y,theta,m));
[theta,J_history] = GradientDescendent(X,Y,theta,m,alpha,iterations);

fprintf('The theta found  by descendant gradient: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,1), theta(1)+(theta(2)*X(:,1)), 'r-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure
fprintf('Program paused. Press enter to continue.\n');
pause;

disp('Show last value of J (min value)'),disp(J_history(iterations));

% Plot the J vs iterations
plot([1:1:iterations],J_history, 'r-')
xlabel('Iterations');
ylabel('J(theta)');
legend('J(theta) vs Iterations ');
