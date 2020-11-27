function [precision, recall] = NNMF(queryNumber)
%NNMF Calculates the Nonnegative Matrix Factorization
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

load 'text-mining-medline_stemmed.mat' A q

k = 50;
[U,S,V] = svd(A,k);

W(:,1) = U(:,1);
for j = 2:k
    C = U(:,j)*V(:,j)';
    C = C.*(C>=0);
    [u,s,v] = svds(C,1);
    W(:,j) = u;  
end 

[Q,R] = qr(W,0);
H = inv(R)*Q'*A;

threshold = 0.5;
diff_H = 1;
diff_W = 1;
while(threshold > diff_H && threshold > diff_W)
    old_H = H;
    old_W = W;
    W = W.*(W>=0);
    H = H.*(W'*V)./((W'*W)*H+epsilon);
    H = H.*(H>=0);
    W = W.*(V*H')./(W*(H*H')+epsilon);
    [W,H] = normalize(W,H);
    diff_H = sum(sum(abs(old_H - H)));
    diff_W = sum(sum(abs(old_W - W)));
end

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


% [W,H] = nnmf(A,k);
% [Q,R] = qr(W,0);
% 
% Rinv = inv(R);
% 
% cosines = zeros(size(q, 2), size(H, 2));
% 
% for queryNum = 1:size(q,2)
%     qHat = Rinv * Q' * q(:,queryNum);
%     
%     for j = 1: size(H, 2)
%         den = normest(qHat) * normest(H(:,j));
%         cosines(queryNum, j) = qHat' * H(:,j) / den;
%     end
% end


end

