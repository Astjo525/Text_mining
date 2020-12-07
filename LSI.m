function [precision, recall] = LSI()
% LSI Calculates the Latent Semantic Indexing
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

load text-mining-medline_stemmed.mat A q

[U,S,V] = svds(A,100);
H = S*V';

qk = U'*q;
%qk = qk(:,queryNumber);
cosines = zeros(1, length(H));

steps = 5:5:90;
vec = zeros(length(steps), size(q,2));

for queryNum = 1:size(q,2)
    qkQuery = qk(:,queryNum);
    for j =1:length(H)
        cosines(:,j) = qkQuery'*H(:,j)/(norm(qkQuery)*norm(H(:,j)));
    end
    [precision, recall]= getPrecisionRecall(cosines, queryNum);
    precision(isnan(precision)) = 0;
    vec(:, queryNum) = interp1q(flip(recall), flip(precision), steps');
end

average_prec = nansum(vec, 2)/sum(~isnan(vec),2);

precision = average_prec;
recall = steps;



%[precision, recall] = getPrecisionRecall(cosines, queryNumber);

% k = 1;
% p = find(abs(U(:,k)) > 0.11);
% words = dict(p,:)

% Multiply by a hundred to get the percent
% precision = precision * 100;
% recall = recall * 100;

end

