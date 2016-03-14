
function [h a_0 a_1] = forwardpropagation(X,w_0,w_1)
    format long;
    m = size(X,1);
    z_0 = X*w_0;
    a_0 = [ ones(m,1) sigmoid(z_0)];
    z_1 = a_0*w_1;
    a_1 = h = sigmoid(z_1);
    %[v,h] = max(a_1,[],2);

end;
