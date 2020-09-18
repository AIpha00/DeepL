# -*- coding: utf-8 -*-
"""
 author: lvsongke@oneniceapp.com
 data:2019/09/11
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import vocabulary
import word2vec_skip

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

plot_only = 1000

##执行降维操作
low_dim_embs = tsne.fit_transform(word2vec_skip.final_embeddings[:plot_only, :])

labels = []

for i in range(plot_only):
    labels.append(vocabulary.reverse_dictionary[i])

plt.figure(figsize=(20, 20))

print('开始画图！....')

for j, label in enumerate(labels):
    x, y = low_dim_embs[j, :]

    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.savefig(fname='after_tsne.png')