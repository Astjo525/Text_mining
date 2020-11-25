function [] = LSI()
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes her
 load text-mining-medline_stemmed.mat A dict q
% path('text-mining-medline.tar/text-mining-medline.tar/MED.REL');
% correctRelDocs = sparse(MED(:,1), MED(:,3), 1) == 1;


[U,Z,V] = svds(A,100);
H = Z*V';
qk = U'*q;
qk = qk(:,9);
costheta = zeros(1,1033);
for j =1:1033
costheta(:,j) = qk'*H(:,j)/(norm(qk)*norm(H(:,j)));
end

costheta = abs(costheta);
% 
% queryNum = 9;
% tols = linspace(0,1);
% precision = zeros(length(tols), 1);
% recall = zeros(length(tols), 1);
% for i = 1:length(tols)
%     modelResult = costheta(1,:) > tols(i);
%     
%     Dr = nnz(modelResult .* correctRelDocs(queryNum,:));
%     Dt = nnz(modelResult);
%     Nr = nnz(correctRelDocs(queryNum,:));
%     
%     precision(i) = Dr / Dt;
%     recall(i) = Dr / Nr;
% end
% 
% plot(recall, precision);
end

