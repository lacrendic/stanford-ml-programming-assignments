function [C, sigma] = paramSearch(X, y, Xval, yval)
    % Search for "optimal" values of C and sigma
    candidates = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    n = length(candidates);
    C = candidates(1);
    sigma = candidates(1);
    min_error = 1;

    for i = 1:n
        for j = 1:n
            cand_c = candidates(i);
            cand_s = candidates(j);

            model = svmTrain(
                X, y, cand_c, @(x1, x2) gaussianKernel(x1, x2, cand_s)); 
            y_pred = svmPredict(model, Xval);

            error_ = mean(double(y_pred ~= yval));
            if error_ < min_error
                C = cand_c
                sigma = cand_s
                min_error = error_
            endif

            fprintf(
                ['C = %f,', 'sigma = %f: ', 'error is %f'], 
                cand_c, cand_s, error_ * 100)
        end
    end

end