clear
queryNumber = 9;

[precision_lsi, recall_lsi] = LSI(queryNumber);

[precision_cl, recall_cl] = Clustering(queryNumber);

[precision_nnmf, recall_nnmf] = NNMF(queryNumber);

plot(recall_lsi, precision_lsi, 'b-d')
hold on
plot(recall_cl, precision_cl, 'k-*')
plot(recall_nnmf, precision_nnmf, 'r-^')
legend('LSI', 'Clustering', 'NNMF')