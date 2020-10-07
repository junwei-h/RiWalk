import os
import sys

ROOT_DIR = os.getcwd()

import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=False)

from gensim.models import KeyedVectors

def plot_projection(embedding_df, hue_on='label', labelsize=20, fontsize=22, labels=[0,1,2],
                         linewidth=0.000001, savefig_path=None):
    
    palette_pool = ['blue','green','red','cyan','magenta','yellow','black']
    markers_pool = ['.','^','X','8','s','p','s','*','+']
    sizes_pool = [30, 100, 10]

    palette = []
    markers = []
    sizes = []
    for i in range(len(labels)):
        palette.append(palette_pool[i%len(palette_pool)])
        markers.append(markers_pool[i%len(markers_pool)])
        sizes.append(sizes_pool[i%len(sizes_pool)])
    # palette=['cadetblue', 'coral', 'brown']
    # markers=['.', 'X', '^']


    fig_dims = (10, 6)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.scatterplot(x='dim_0', y='dim_1', hue=hue_on, style=hue_on, markers=markers, size=hue_on,
                    sizes=sizes, linewidth=linewidth, palette=palette, data=embedding_df, ax=ax)
    ax.set_xlabel('Dimension 1', fontsize=fontsize)
    ax.set_ylabel('Dimension 2', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
    L = ax.legend(prop={'size': labelsize}, facecolor="#EEEEEE", handletextpad=-0.5)
    if hue_on == 'label':
        L.get_texts()[0].set_text('True label')
    else:
        L.get_texts()[0].set_text('Predicted label')
    
    for i in range(len(labels)):
        L.get_texts()[i+1].set_text(u"Label {}".format(i))

    if savefig_path == None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)
    fig.show()

def save_as_umap(file_name):
    # load emb file
    wv = KeyedVectors.load_word2vec_format(os.path.join(ROOT_DIR,u'embs/{}.emb'.format(file_name)), binary=False)
    wv_df = pd.DataFrame(wv.vectors,index=[int(i) for i in wv.index2word])
    df = pd.read_csv(os.path.join(ROOT_DIR, u'graphs/{}.group'.format(file_name)), header=None, delimiter=" ", names=["label"], index_col=0)
    df_label = wv_df.join(df)['label']
    unique_labels = sorted(df_label.unique())

    # # normalize features
    # for col in wv_df.columns:
    #     x = np.mean(wv_df[col])
    #     s = np.std(wv_df[col])
    #     wv_df[col] = (wv_df[col] - x)/s 
    # # preprocess by PCA 
    # pca = PCA(random_state=42, n_components=20)
    # wv_pca_df = pca.fit_transform(wv_df)

    # reducer = umap.UMAP(random_state=42, n_components=2, min_dist=0.01, n_neighbors=15, learning_rate=0.1)
    # embedding = reducer.fit_transform(wv_df)

    reducer = TSNE(random_state=42, n_components=2, perplexity=30, n_iter=1000, learning_rate=100.0)
    embedding = reducer.fit_transform(wv_df)  

    embedding_df = pd.DataFrame(embedding, columns=('dim_0', 'dim_1'))
    embedding_df['label'] = df_label.tolist()

    plot_projection(embedding_df=embedding_df, hue_on='label', fontsize=19, labelsize=22,
                     labels=unique_labels
                     #savefig_path=os.path.join(ROOT_DIR, u'figures/{}_umap_true_label.png'.format(file_name))
                     )

def gen_struc_features(file_prefix, t_list):
    # load emb file

    df_list = []
    for t in t_list:
        file_name = file_prefix+u'{}'.format(t)
        print("reading {}".format(file_name))
        wv = KeyedVectors.load_word2vec_format(os.path.join(ROOT_DIR,u'embs/{}.emb'.format(file_name)), binary=False)
        wv_df = pd.DataFrame(wv.vectors,index=[int(i) for i in wv.index2word])

        df_list.append(wv_df)
    
    df_emb = pd.concat(df_list)

    for col in df_emb.columns:
        x = np.mean(df_emb[col])
        s = np.std(df_emb[col])
        df_emb[col] = (df_emb[col] - x)/s 

    df_emb.to_csv(os.path.join(ROOT_DIR, 'embs/elliptic_struc_features.csv'),header=None)


def main():
    #save as umap figure
    # file_name = "mykarate"
    # save_as_umap(file_name) 

    #merge emb as new features
    file_prefix = "elliptic_t"
    num_list = range(1,50)
    gen_struc_features(file_prefix, num_list)


if __name__ == '__main__':
    main()