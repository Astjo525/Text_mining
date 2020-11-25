function [precision, recall] = getPrecisionRecall(cosines, queryNumber)
% GETPRECISIONRECALL Return the precision and recall for the query
% determined by the queryNumber,
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

load text-mining-medline.tar/MED.REL MED

index = MED(:, 1) == queryNumber;
relevant_docs = MED(index, 3);

tol = linspace(0,0.98);
precision = zeros(length(tol),1);
recall = zeros(length(tol),1);

for n = 1:length(tol)
    retrieved_docs_sparse = (cosines(1,:) > tol(n));  % retrieved_docs_sparse = (cosines(1,:) > tol(n));
    [~, retrieved_docs, ~] = find(retrieved_docs_sparse);
    retrieved_docs = retrieved_docs';

    retrieved_relevant_docs = sum(ismember(retrieved_docs, relevant_docs));
    precision(n) = retrieved_relevant_docs / length(retrieved_docs);

    recall(n) = retrieved_relevant_docs / length(relevant_docs);
end

% Multiply by a hundred to get the percent
precision = precision * 100;
recall = recall * 100;

% Algot's alternative solution
% TODO: Delete, probably
% correctRelDocs = sparse(MED(:,1), MED(:,3), 1) == 1;
% 
% tols = linspace(0,0.98, 100);
% precision = zeros(length(tols), 1);
% recall = zeros(length(tols), 1);
% for i = 1:length(tols)
%     modelResult = cosines(1,:) > tols(i);
%     
%     Dr = nnz(modelResult .* correctRelDocs(queryNumber,:));
%     Dt = nnz(modelResult);
%     Nr = nnz(correctRelDocs(queryNumber,:));
%     
%     precision(i) = Dr / Dt;
%     recall(i) = Dr / Nr;
% end
% 
% 
% % Multiply by a hundred to get the percent
% precision = precision * 100;
% recall = recall * 100;

end

