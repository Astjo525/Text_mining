function [precision, recall] = NNMF(queryNumber)
%NNMF Calculates the Nonnegative Matrix Factorization
%   queryNumber is the number of the query to compare
%   precision and recall is returned in percent (0-100)

load 'text-mining-medline_stemmed.mat' A q

k = 50;

% Calculate W
[U,S,V] = svds(A,k);
W(:,1) = U(:,1);
for j = 2:k
    C = U(:,j)*V(:,j)';
    C = C.*(C>=0);
    [u,s,v] = svds(C,1);
    W(:,j) = u;  
end 

% Calculate H
[Q,R] = qr(W,0);
H = inv(R)*Q'*A;

% Iteratively improve W and H
epsilon = 1e-10;
for i = 1:100
    W = W.*(W>=0);
    H = H.*(W'*A)./((W'*W)*H+epsilon);
    H = H.*(H>=0);
    W = W.*(A*H')./(W*(H*H')+epsilon);
    W = normalize(W);
    H = normalize(H);
end

%%% ITERATION ACCORDING TO BOOK. IMPROVE THIS INSTEAD OF ABOVE FOR-LOOP %%%
% threshold = 0.5;
% diff_H = 1;
% diff_W = 1;
% it = 0;
% while(threshold > diff_H && threshold > diff_W)
%     old_H = H;
%     old_W = W;
%     W = W.*(W>=0);
%     H = H.*(W'*V)./((W'*W)*H+epsilon);
%     H = H.*(H>=0);
%     W = W.*(V*H')./(W*(H*H')+epsilon);
%     [W,H] = normalize(W,H);
%     diff_H = sum(sum(abs(old_H - H)));
%     diff_W = sum(sum(abs(old_W - W)));
%     it = it + 1;
% end

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

[precision, recall] = getPrecisionRecall(cosines(queryNumber,:), queryNumber);

end

