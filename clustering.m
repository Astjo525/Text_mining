clear
load('text-mining-medline_stemmed.mat');
load('text-mining-medline.tar/MED.REL');
correctRelDocs = sparse(MED(:,1), MED(:,3), 1) == 1; % hehe

k = 50;

% A = normc(A);

[~, Ck] = kmeans(A', k);
Ck = Ck';

[Pk, R] = qr(Ck, 0);

Gk = Pk' * A;

qk = Pk' * q;

cosines = zeros(size(q, 2), size(Gk, 2));
for queryNum = 1:size(q,2)
    for j = 1: size(Gk, 2)
        den = normest(qk(:,queryNum)) * normest(Gk(:,j));
        cosines(queryNum, j) = qk(:,queryNum)' * Gk(:,j) / den;
    end
end

%%
queryNum = 9;

tols = linspace(0,0.98, 100);
precision = zeros(length(tols), 1);
recall = zeros(length(tols), 1);
for i = 1:length(tols)
    modelResult = cosines(queryNum,:) > tols(i);
    
    Dr = nnz(modelResult .* correctRelDocs(queryNum,:));
    Dt = nnz(modelResult);
    Nr = nnz(correctRelDocs(queryNum,:));
    
    precision(i) = Dr / Dt;
    recall(i) = Dr / Nr;
end

plot(recall * 100, precision * 100, '-o');


%%
% query_number = 9;
% 
% ind = MED(:, 1) == query_number;
% relevant_docs = MED(ind, 3);
% 
% tol = linspace(0,1, 100);
% precision = zeros(length(tol),1);
% recall = zeros(length(tol),1);
% 
% for n = 1:length(tol)
%     retrieved_docs_sparse = (cosines(query_number, :) > tol(n));
%     [~, retrieved_docs, ~] = find(retrieved_docs_sparse);
%     retrieved_docs = retrieved_docs';
% 
%     retrieved_relevant_docs = sum(ismember(retrieved_docs, relevant_docs));
%     precision(n) = retrieved_relevant_docs / length(retrieved_docs);
% 
%     recall(n) = retrieved_relevant_docs / length(relevant_docs);
% end
% 
% %figure
% plot(recall(:)*100, precision(:)*100, '*')


%%
% Precision

% Normalize columns before clustering?


