function [precision, recall] = NNMF()
%NNMF Calculates the Nonnegative Matrix Factorization method for query
% matching in text mining
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

load 'text-mining-medline_stemmed.mat' A q

% Set k rank and threshold
k = 50;
threshold = 0.01;

% Compute the nonnegative matrix factorization of A
[W, H] = NNMFAlgorithm(A, k, threshold);

[Q,R] = qr(W,0);

qHat = R\Q' * q;

cosines = zeros(size(q, 2), size(H, 2));

steps = 5:5:90;
vec = zeros(length(steps), size(q,2));

for queryNum = 1:size(q,2)
    %qHat = R\Q' * q(:,queryNum);
    
    for j = 1: size(H, 2)
        den = normest(qHat(:,queryNum)) * normest(H(:,j));
        cosines(:, j) = qHat(:, queryNum)' * H(:,j) / den;
    end
    [precision, recall] = getPrecisionRecall(cosines, queryNum);
    precision(isnan(precision)) = 0;
    vec(:, queryNum) = interp1q(flip(recall), flip(precision), steps');

end

average_prec = nansum(vec, 2)/sum(~isnan(vec),2);

precision = average_prec;
recall = steps;
%[precision, recall] = getPrecisionRecall(cosines(queryNumber,:), queryNumber);

end

