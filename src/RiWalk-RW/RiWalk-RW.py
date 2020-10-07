# -*- coding: utf-8 -*-
"""
Reference implementation of RiWalk.
Author: Xuewei Ma
For more details, refer to the paper:
RiWalk: Fast Structural Node Embedding via Role Identification
ICDM, 2019
"""

import argparse
import json
import time
import RiWalkRWGraph
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
import networkx as nx
import os
import glob
import logging
import sys
 
#below added to view embedding 2-dim
import umap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
rc('text', usetex=False)
ROOT_DIR = os.getcwd()

def plot_UMAP_projection(embedding_df, hue_on='label', labelsize=20, fontsize=22, labels=[0,1,2],
                         linewidth=0.000001, savefig_path=None):
    
    palette_pool = ['blue','green','red','cyan','magenta','yellow','black']
    markers_pool = ['.','^','X','8','s','p','s','*','+']
    sizes_pool = [30, 150, 10]

    palette = []
    markers = []
    sizes = []
    for i in range(len(labels)):
        palette.append(palette_pool[i%len(palette_pool)])
        markers.append(markers_pool[i%len(markers_pool)])
        sizes.append(sizes_pool[i%len(sizes_pool)])

    fig_dims = (20, 12)
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
        L.get_texts()[i+1].set_text(u"Label {}".format(labels[i]))

    if savefig_path == None:
        plt.show()
    else:
        plt.savefig(savefig_path, bbox_inches='tight', pad_inches=0)
    fig.show()

def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def parse_args():
    """
        Parses the RiWalk arguments.
    """
    parser = argparse.ArgumentParser(description="Run RiWalk")

    parser.add_argument('--input', nargs='?', default='graphs/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='embs/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=80,
                        help='Number of walks per source. Default is 80.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD. Default is 5.')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 4.')

    parser.add_argument('--flag', nargs='?', default='sp',
                        help='Flag indicating using RiWalk-SP(sp) or RiWalk-WL(wl). Default is sp.')

    parser.add_argument('--without-discount', action='store_true', default=False,
                        help='Flag indicating not using discount.')

    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument("-l", "--log", dest="log", default="DEBUG",
                        help="Log verbosity level. Default is DEBUG.")

    return parser.parse_args()


class Sentences(object):
    """
    a wrapper of random walk files to feed to word2vec
    """
    def __init__(self, file_names):
        self.file_names = file_names

    def __iter__(self):
        fs = []
        for file_name in self.file_names:
            fs.append(open(file_name))
        while True:
            flag = 0
            for i, f in enumerate(fs):
                line = f.readline()
                if line != '':
                    flag = 1
                    yield line.split()
            if not flag:
                try:
                    for f in fs:
                        f.close()
                except:
                    pass
                return


class RiWalk:
    def __init__(self, args):
        self.args = args
        os.system('rm -rf walks/__random_walks_*.txt')

    def learn_embeddings(self):
        """
        learn embeddings from random walks.
        hs:  0:negative sampling 1:hierarchica softmax
        sg:  0:CBOW              1:skip-gram
        """
        dim = self.args.dimensions
        window_size = self.args.window_size
        workers = self.args.workers
        iter_num = self.args.iter

        logging.debug('begin learning embeddings')
        learning_begin_time = time.time()

        walk_files = glob.glob('walks/__random_walks_*.txt')
        sentences = Sentences(walk_files)
        model = Word2Vec(sentences, size=dim, window=window_size, min_count=0, sg=1, hs=0, workers=workers, iter=iter_num, seed=9900)

        learning_end_time = time.time()
        logging.debug('done learning embeddings')
        logging.debug('learning_time: {}'.format(learning_end_time - learning_begin_time))
        print('learning_time', learning_end_time - learning_begin_time, flush=True)
        return model.wv

    def read_graph(self):
        logging.debug('begin reading graph')
        read_begin_time = time.time()

        input_file_name = self.args.input
        nx_g = nx.read_edgelist(input_file_name, nodetype=int, create_using=nx.DiGraph())
        for edge in nx_g.edges():
            nx_g[edge[0]][edge[1]]['weight'] = 1
        # nx_g = nx_g.to_undirected() # an edge [0,1] will be returned to [[0,1],[1,0]]
        nx_g = nx_g.to_directed() #Test

        logging.debug('done reading graph')
        read_end_time = time.time()
        logging.debug('read_time: {}'.format(read_end_time - read_begin_time))
        return nx_g

    def preprocess_graph(self, nx_g):
        """
        1. relabel nodes with 0,1,2,3,...,N.
        2. convert graph to adjacency representation as a list of tuples.
        """
        logging.debug('begin preprocessing graph')
        preprocess_begin_time = time.time()

        mapping = {_: i for i, _ in enumerate(nx_g.nodes())}
        nx_g = nx.relabel_nodes(nx_g, mapping)
        nx_g = [tuple(nx_g.neighbors(_)) for _ in range(len(nx_g))]

        logging.info('#nodes: {}'.format(len(nx_g)))
        logging.info('#edges: {}'.format(sum([len(_) for _ in nx_g]) // 2))

        logging.debug('done preprocessing')
        logging.debug('preprocess time: {}'.format(time.time() - preprocess_begin_time))
        return nx_g, mapping

    def learn(self, nx_g, mapping):
        g = RiWalkRWGraph.RiGraph(nx_g, self.args)

        walk_time, bfs_time, ri_time, walks_writing_time = g.process_random_walks()

        print('walk_time', walk_time / self.args.workers, flush=True)
        print('bfs_time', bfs_time / self.args.workers, flush=True)
        print('ri_time', ri_time / self.args.workers, flush=True)
        print('walks_writing_time', walks_writing_time / self.args.workers, flush=True)
        
        wv = self.learn_embeddings()

        original_wv = Word2VecKeyedVectors(self.args.dimensions)
        original_nodes = list(mapping.keys())
        original_vecs = [wv.word_vec(str(mapping[node])) for node in original_nodes]
        original_wv.add(entities=list(map(str, original_nodes)), weights=original_vecs)
        return original_wv

    def riwalk(self):
        nx_g = self.read_graph()
        read_end_time = time.time()
        nx_g, mapping = self.preprocess_graph(nx_g)
        wv = self.learn(nx_g, mapping)
        return wv, time.time() - read_end_time


def main():
    args = parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

    os.system('rm -f RiWalk.log')
    logging.basicConfig(filename='RiWalk.log', level=numeric_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    logging.info(str(vars(args)))
    if args.debug:
        sys.excepthook = debug

    wv, total_time = RiWalk(args).riwalk()

    write_begin_time = time.time()
    wv.save_word2vec_format(fname=args.output, binary=False)
    logging.debug('embedding_writing_time: {}'.format(time.time() - write_begin_time))

    json.dump({'time': total_time}, open(args.output.replace('.emb', '_time.json'), 'w'))


    # visualizes embedding
    wv_df = pd.DataFrame(wv.vectors,index=[int(i) for i in wv.index2word])
    df = pd.read_csv(os.path.join(ROOT_DIR, args.input.replace('.edgelist', '.group')), header=None, delimiter=" ", names=["label"], index_col=0)
    df_label = wv_df.join(df)['label']
    unique_labels = sorted(df_label.unique())

    reducer = umap.UMAP(random_state=42, n_components=2, min_dist=0.01, n_neighbors=15, learning_rate=0.1)
    embedding = reducer.fit_transform(wv_df)

    embedding_df = pd.DataFrame(embedding, columns=('dim_0', 'dim_1'))
    embedding_df['label'] = df_label.tolist()

    plot_UMAP_projection(embedding_df=embedding_df, hue_on='label', fontsize=19, labelsize=22,
        labels=unique_labels,
        savefig_path=os.path.join(ROOT_DIR, u'figures/{}_umap_iter{}_flag{}_size{}.png'.format(args.input.split('/')[1].split('.')[0],args.iter, args.flag, args.window_size))
        )


if __name__ == '__main__':
    main()
