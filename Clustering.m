function [precision, recall] = Clustering()
% Clustering Clustering is done with the k-means method
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

load 'text-mining-medline_stemmed.mat' A q

k = 100;

A = normc(A);

[~, Ck] = kmeans(A', k, 'MaxIter', 2000);
Ck = Ck';

[Pk, ~] = qr(Ck, 0);

Gk = Pk' * A;

qk = Pk' * q;

% Matilda's method
% TODO: delete, probably
% for n = 1:length(Gk)
%     cosines_matilda(:,n) = (qk'*Gk(:,n)) / (normest(qk,2) * normest(Gk(:,n),2));
% end

cosines = zeros(size(q, 2), size(Gk, 2));

steps = 5:5:90;
vec = zeros(length(steps), size(q,2));

for queryNum = 1:size(q,2)
    for j = 1: size(Gk, 2)
        den = normest(qk(:,queryNum)) * normest(Gk(:,j));
        cosines(:, j) = qk(:,queryNum)' * Gk(:,j) / den;
    end
    [precision, recall] = getPrecisionRecall(cosines, queryNum);
    precision(isnan(precision)) = 0;
    vec(:, queryNum) = interp1q(flip(recall), flip(precision), steps');
end

average = nansum(vec, 2)/sum(~isnan(vec),2);
average(:,all(average == 0))=[];

precision = average;
recall = steps;


%[precision, recall] = getPrecisionRecall(cosines(queryNumber,:), queryNumber);

% Multiply by a hundred to get the percent
% precision = precision * 100;
% recall = recall * 100;

% TODO: Delete, probably
%[precision_m, recall_m] = getPrecisionRecall(cosines_matilda(queryNumber,:), queryNumber);
% precision_m = precision_m * 100;
% recall_m = recall_m * 100;

% plot(recall, precision, 'b-o')
% hold on
% plot(recall_m, precision_m, 'r-*')
% legend('Algot', 'Matilda')

end

