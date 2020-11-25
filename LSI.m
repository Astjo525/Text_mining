function [precision, recall] = LSI(queryNumber)
% LSI Calculates the Latent Semantic Indexing
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

load text-mining-medline_stemmed.mat A q

[U,Z,V] = svds(A,100);
H = Z*V';
qk = U'*q;
qk = qk(:,queryNumber);
cosines = zeros(1, length(H));

for j =1:length(H)
    cosines(:,j) = qk'*H(:,j)/(norm(qk)*norm(H(:,j)));
end

cosines = abs(cosines);

[precision, recall] = getPrecisionRecall(cosines, queryNumber);

% Multiply by a hundred to get the percent
% precision = precision * 100;
% recall = recall * 100;

end

