clear
load('text-mining-medline_stemmed.mat');
load('text-mining-medline.tar/MED.REL');
correctRelDocs = sparse(MED(:,1), MED(:,3), 1) == 1; % hehe

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