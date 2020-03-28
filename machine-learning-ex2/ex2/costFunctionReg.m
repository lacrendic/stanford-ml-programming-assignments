function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i = 1:m
        x_i = X(i, :);
        y_i = y(i);
        h_i = predict(x_i, theta);
        J = J - y_i * log(h_i) - (1 - y_i) * log(1 - h_i);
        for j = 1:length(grad)
            grad(j) = grad(j) + (h_i - y_i) * x_i(j);
        end
    end

    J = 1/m * J + lambda / (2 * m) * theta(2:n)' * theta(2:n);
    grad = 1/m * grad;
    for j = 2:n
        grad(j) = grad(j) + lambda / m * theta(j);
    end

% =============================================================

end



function h = predict(x, theta)
    h = sigmoid(x * theta);
end