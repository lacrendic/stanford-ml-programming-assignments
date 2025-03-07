function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    %NNCOSTFUNCTION Implements the neural network cost function for a two layer
    %neural network which performs classification
    %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    %   X, y, lambda) computes the cost and gradient of the neural network. The
    %   parameters for the neural network are "unrolled" into the vector
    %   nn_params and need to be converted back into the weight matrices. 
    % 
    %   The returned parameter grad should be a "unrolled" vector of the
    %   partial derivatives of the neural network.
    %

    % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    % for our 2 layer neural network
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                    num_labels, (hidden_layer_size + 1));

    % Setup some useful variables
    m = size(X, 1);
            
    % You need to return the following variables correctly 
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));

    % ====================== YOUR CODE HERE ======================
    % Instructions: You should complete the code by working through the
    %               following parts.
    %
    % Part 1: Feedforward the neural network and return the cost in the
    %         variable J. After implementing Part 1, you can verify that your
    %         cost function computation is correct by verifying the cost
    %         computed in ex4.m
    %
    % Part 2: Implement the backpropagation algorithm to compute the gradients
    %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    %         Theta2_grad, respectively. After implementing Part 2, you can check
    %         that your implementation is correct by running checkNNGradients
    %
    %         Note: The vector y passed into the function is a vector of labels
    %               containing values from 1..K. You need to map this vector into a 
    %               binary vector of 1's and 0's to be used with the neural network
    %               cost function.
    %
    %         Hint: We recommend implementing backpropagation using a for-loop
    %               over the training examples if you are implementing it for the 
    %               first time.
    %
    % Part 3: Implement regularization with the cost function and gradients.
    %
    %         Hint: You can implement this around the code for
    %               backpropagation. That is, you can compute the gradients for
    %               the regularization separately and then add them to Theta1_grad
    %               and Theta2_grad from Part 2.
    %

    X = [ones(m, 1) X];

    for i = 1:m
        train_i_cost = 0;

        # Predict intermediate and final values
        [a1 z2 a2 z3 h] = predict(X(i, :)', Theta1, Theta2);

        # One-hot encode y(i)
        y_i = zeros(num_labels, 1);
        y_i(y(i)) = 1;

        # Update cost function
        for k = 1:num_labels
            # Computing right value of y_k^(i)
            y_k = y_i(k);

            # Computing local cost
            k_cost = -1 * y_k * log(h(k)) - (1 - y_k) * log(1 - h(k));

            # Adding to total cost
            train_i_cost += k_cost;
        end
        J += train_i_cost;

        # Update gradients
        # Step 1: compute difference for output layer
        delta_3 = h - y_i;

        # Step 2: compute difference for hidden layer
        delta_2 = (Theta2' * delta_3);
        delta_2 = delta_2(2:end);
        delta_2 = delta_2 .* sigmoidGradient(z2);

        # Step 3: compute gradient
        Theta1_grad += delta_2 * a1';
        Theta2_grad += delta_3 * a2';
    end

    J /= m;
    Theta1_grad /= m;
    Theta2_grad /= m;

    # Add regularization cost
    reg_theta_1_cost = norm(Theta1(:, 2:end), p='fro') ^ 2;
    reg_theta_2_cost = norm(Theta2(:, 2:end), p='fro') ^ 2;
    J += lambda / (2*m) * (reg_theta_1_cost + reg_theta_2_cost);

    # Add regularization term to gradients
    Theta1_grad += lambda / m * [zeros(size(Theta1, 1), 1), Theta1(:, 2:end)];
    Theta2_grad += lambda / m * [zeros(size(Theta2, 1), 1), Theta2(:, 2:end)];


    % -------------------------------------------------------------

    % =========================================================================

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


function [a1 z2 a2 z3 h] = predict(x, Theta1, Theta2)
    a1 = x;
    z2 = Theta1 * x;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    h = sigmoid(z3);
end
