function [W, H] = NNMFAlgorithm(A, k, threshold)
%NNMF Calculates an approximation of a Nonnegative Matrix Factorization
%   A is the term document matrix
%   W, H are the computed nnmf matrices  

% Initialize W
[U,~,V] = svds(A,k);
W(:,1) = U(:,1);
for j = 2:k
    C = U(:,j)*V(:,j)';
    C = C.*(C>=0);
    [u,~,~] = svds(C,1);
    W(:,j) = abs(u); 
end 

% Calculate H
[Q,R] = qr(W,0);
H = R\Q'*A;  

% Iteratively change W and H until threshold reached
W_norm = threshold +1;
it = 0;
epsilon = 1e-5;
while(threshold < W_norm)
    prev_W = W;
    
    % Remove negativa values from W and H and update matrices values
    W = W.*(W>=0);
    H = H.*(W'*A)./((W'*W)*H+epsilon);
    H = H.*(H>=0);
    W = W.*(A*H')./(W*(H*H')+epsilon);
    
    % Normalize W and update H accordingly
    c = norm(W, 'fro');
    W = W/c;
    H = c*H;
    
    % Calculate how much W changed from last iteration
    W_norm = norm(prev_W - W,'fro');
    
    % Change in W to be plotted
    it = it + 1;
    W_norm_plot(1,it) = W_norm;
end

% Plot change in W to find where it converges
x_range = 1:length(W_norm_plot);
figure
plot(x_range, W_norm_plot);

end

