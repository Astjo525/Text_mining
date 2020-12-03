function [precision, recall] = NNMF(queryNumber)
%NNMF Calculates the Nonnegative Matrix Factorization method for query
% matching in text mining
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

load 'text-mining-medline_stemmed.mat' A q

% Set k rank and threshold
k = 50;
threshold = 0.1

% Compute the nonnegative matrix factorization of A
[W, H] = NNMFAlgorithm(A, k, threshold);

[Q,R] = qr(W,0);

cosines = zeros(size(q, 2), size(H, 2));

for queryNum = 1:size(q,2)
    qHat = R\Q' * q(:,queryNum);
    
    for j = 1: size(H, 2)
        den = normest(qHat) * normest(H(:,j));
        cosines(queryNum, j) = qHat' * H(:,j) / den;
    end
end

[precision, recall] = getPrecisionRecall(cosines(queryNumber,:), queryNumber);

end

