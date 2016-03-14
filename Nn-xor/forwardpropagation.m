
function [h a_2 a_3] = forwardpropagation(X,w_2,w_3,m)
    z_2 = X * w_2;
    a_2 = [ ones(m,1) sigmoid(z_2)];
    z_3 = a_2 * w_3;
    a_3 = h = sigmoid(z_3);
end;
