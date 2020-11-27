function [recall,precision] = NNMFMatlab(queryNumber)
%NNMF Calculates the Nonnegative Matrix Factorization with Matlab's
%   function nnmf
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

    load 'text-mining-medline_stemmed.mat' A q

    k = 50;

    [W,H] = nnmf(A,k);
    [Q,R] = qr(W,0);

    Rinv = inv(R);

    cosines = zeros(size(q, 2), size(H, 2));

    for queryNum = 1:size(q,2)
        qHat = Rinv * Q' * q(:,queryNum);

        for j = 1: size(H, 2)
            den = normest(qHat) * normest(H(:,j));
            cosines(queryNum, j) = qHat' * H(:,j) / den;
        end
    end
    
    [precision, recall] = getPrecisionRecall(cosines(queryNumber,:), queryNumber);
    
end

