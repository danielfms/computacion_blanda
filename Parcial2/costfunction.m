function J = costfunction(Y,h,m)

% Vectorize cost function

    J=(0.5)*sum(( Y - h ).^2);

end
