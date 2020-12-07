clear

[precision_lsi, recall_lsi] = LSI();
[precision_cl, recall_cl] = Clustering();
[precision_nnmf, recall_nnmf] = NNMF();

figure
plot(recall_lsi, precision_lsi, '-d')
hold on
plot(recall_cl, precision_cl, '-*')
plot(recall_nnmf, precision_nnmf, '-^')
legend('LSI', 'Clustering', 'NNMF')
xlabel('Recall (%)')
ylabel('Average Precision (%)')
title('Average')
