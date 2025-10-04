'''
This file provides implementation of Word2Vec skip grams and CBOW architecture using negative sampling.
Both skig gram and CBOW are implemented as separate classes.
outputs for each architecture:
1. embedding matrix
2. t-SNE and umap file visualising embeddings in 2d space

The learned embeddings can be used for analogy tasks, such as:
king – man + woman → queen
'''


!pip install datasets


import nltk
import os
from datasets import load_dataset
import numpy as np
import re
import random
from collections import Counter
from nltk.tokenize import word_tokenize

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import pickle
#from google.colab import files

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
HF_TOKEN="insert your hugging face token here"


# Loading Data
def load_hf_data():
    print("Loading and tokenizing dataset...")
    ds = load_dataset("dataset", "subset",token=HF_TOKEN, split="train")
    text_data = ds["text"]
    return [word_tokenize(doc.strip().lower()) for doc in text_data if doc.strip()]


data = load_hf_data()
def pick_random_words(word2idx, k=50, seed=42):
    special_tokens = {"<unk>", "<pad>", "<s>", "</s>"}
    vocab_words = [w for w in word2idx.keys() if w not in special_tokens]
    rng = random.Random(seed)
    return rng.sample(vocab_words, k)

def top_k_similar(word, word2idx, idx2word, embeddings, top_k=10):
    center_idx = word2idx[word]
    v = embeddings[center_idx].reshape(1, -1)
    sims = cosine_similarity(v, embeddings).flatten()
    sims[center_idx] = -np.inf
    top_indices = np.argpartition(-sims, range(top_k))[:top_k]
    top_indices = top_indices[np.argsort(-sims[top_indices])]
    return [(idx, idx2word[idx], sims[idx]) for idx in top_indices]


def collect_neighbor_set(sampled_words, word2idx, idx2word, embeddings, top_k=10):
    words = []
    emb_rows = []
    meta = []
    for q in sampled_words:
        neighbors = top_k_similar(q, word2idx, idx2word, embeddings, top_k=top_k)
        for idx, neigh_word, sim in neighbors:
            words.append(neigh_word)
            emb_rows.append(embeddings[idx])
            meta.append({"query": q, "neighbor": neigh_word, "sim": float(sim)})

    emb_array = np.vstack(emb_rows)
    return words, emb_array, meta


def reduce_and_plot(emb_array, labels, meta, method="tsne", title="tsne plot",tsne_params=None, umap_params=None, random_state=42):
    if method == "tsne":
        if tsne_params is None:
            tsne_params = {"n_components": 2, "perplexity": 30, "random_state": random_state, "init": "pca", "learning_rate": "auto"}
        reducer = TSNE(**tsne_params)
    else:
        if umap_params is None:
            umap_params = {"n_components": 2, "random_state": random_state}
        reducer = umap.UMAP(**umap_params)

    reduced = reducer.fit_transform(emb_array)
    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "word": labels,
        "query_word": [m["query"] for m in meta],
        "sim": [m["sim"] for m in meta]
    })

    df["hover"] = df.apply(lambda r: f"word: {r['word']}<br>query: {r['query_word']}<br>sim: {r['sim']:.4f}", axis=1)

    fig = px.scatter(df, x="x", y="y", hover_name="word", hover_data={"query_word": True, "sim": True, "x": False, "y": False}, title=method+" "+title)
    return fig


def produce_neighbor_visualizations(embeddings, word2idx, idx2word,title,
                                    sample_k=50, top_k=10, seed=42):
    sampled_words = pick_random_words(word2idx, k=sample_k, seed=seed)
    words_list, emb_array, meta = collect_neighbor_set(sampled_words, word2idx, idx2word, embeddings, top_k=top_k)

    fig_tsne = reduce_and_plot(emb_array, words_list, meta, method="tsne", title=title, random_state=seed)
    fig_umap = reduce_and_plot(emb_array, words_list, meta, method="umap", title=title, random_state=seed)

    return {
        "tsne": fig_tsne,
        "umap": fig_umap,
    }

def analogy(a, b, c, embeddings, word2idx, idx2word):
    E = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    try:
      vec = E[word2idx[a]] + E[word2idx[b]] - E[word2idx[c]]
    except:
      return np.nan

    vec /= np.linalg.norm(vec)

    sims = E @ vec

    for w in [a, b, c]:
        sims[word2idx[w]] = -np.inf

    top_idx = np.argsort(-sims)[:1]
    return idx2word[top_idx[0]]
def build_vocab(tokens,freq):
    token_freq = Counter(tokens)
    token_freq = {w: c for w, c in token_freq.items() if c >= freq}
    sorted_words = sorted(token_freq.items(), key=lambda x: -x[1])
    word2idx = {w: i for i, (w, _) in enumerate(sorted_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    counts = np.array([c for _, c in sorted_words], dtype=np.float32)
    return word2idx, idx2word, counts

def encode(tokens, word2idx):
    return [word2idx[w] for w in tokens if w in word2idx]


def generate_skipgram_pairs(sentences, window_size):
    pairs = []
    for sentence in sentences:
        length = len(sentence)
        for center_pos, center_word in enumerate(sentence):
            start = max(0, center_pos - window_size)
            end = min(length, center_pos + window_size + 1)
            for context_pos in range(start, end):
                if context_pos != center_pos:
                    pairs.append((center_word, sentence[context_pos]))
    return pairs
'''
Implementation of skip grams architecture with negative sampling
'''

class skip_grams_with_neg_sampling:
    def __init__(self, vocab_size, embed_dim=100, learning_rate=0.05, neg_count=5, table_size=int(1e7)):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = learning_rate
        self.neg_count = neg_count

        self.W_in = np.random.uniform(-0.5/self.embed_dim, 0.5/self.embed_dim, (vocab_size, embed_dim)).astype(np.float32)
        self.W_out = np.random.uniform(-0.5/self.embed_dim, 0.5/self.embed_dim, (vocab_size, embed_dim)).astype(np.float32)


        self.unigram_table = None
        self.table_size=0

    def build_unigram_table(self,counts):
      scaled_counts = counts ** 0.75
      repeats = np.round(scaled_counts).astype(int)  # how many times each word index appears
      table = np.repeat(np.arange(len(counts)), repeats)
      self.table_size=len(table)
      return table.astype(np.int32)

    def build_unigram(self, counts):
        self.unigram_table = self.build_unigram_table(counts)



    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, pairs, epochs=5, batch_size=1024,a=None,b=None):
        for epoch in range(epochs):
            random.seed(42 + epoch)
            random.shuffle(pairs)

            batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
            total_loss = 0
            for centers_contexts in batches:
                centers, contexts = zip(*centers_contexts)
                centers = np.array(centers, dtype=np.int32)
                contexts = np.array(contexts, dtype=np.int32)

                loss = self.train_batch(centers, contexts)
                total_loss += loss * len(centers)

            print(f"Epoch {epoch+1}, Loss={total_loss/len(pairs):.4f}")

    def train_batch(self, centers, contexts):
        batch_size = len(centers)



        V_c = self.W_in[centers]
        U_o = self.W_out[contexts]

        scores_pos = np.sum(V_c * U_o, axis=1)
        p_pos = self.sigmoid(scores_pos)
        loss = -np.sum(np.log(p_pos + 1e-100))

        grad_pos = self.lr * (1 - p_pos)[:, None]
        self.W_out[contexts] += grad_pos * V_c
        grad_v = grad_pos * U_o

        neg_samples = np.random.randint(self.table_size, size=(batch_size, self.neg_count)) # very small chance of positive sample getting sampled as negative sample
        neg_samples = self.unigram_table[neg_samples]

        U_neg = self.W_out[neg_samples]
        scores_neg = np.einsum('ij,ikj->ik', V_c, U_neg)  # dot product between each center vector and each of its neg_count negative samples
        p_neg = 1 / (1 + np.exp(scores_neg))

        loss -= np.sum(np.log(p_neg + 1e-100))

        grad_neg = self.lr * (1 - p_neg)
        self.W_out[neg_samples] -= grad_neg[:, :, None] * V_c[:, None, :]
        grad_v -= np.sum(grad_neg[:, :, None] * U_neg, axis=1)

        self.W_in[centers] += grad_v

        return loss / batch_size

flat_tokens = [w for sent in data for w in sent]
word2idx, idx2word, counts = build_vocab(flat_tokens,5)

encoded_sentences = [[word2idx[w] for w in sent if w in word2idx] for sent in data]
pairs = generate_skipgram_pairs(encoded_sentences, window_size=2)

skipgram_ns_ = skip_grams_with_neg_sampling(len(word2idx), embed_dim=100, learning_rate=0.01, neg_count=5)
skipgram_ns_.build_unigram(counts)
skipgram_ns_.train(pairs, epochs=10, batch_size=32,a=word2idx,b=idx2word)

skipgram_ns={}
skipgram_ns["embeddings"]= skipgram_ns_.W_in
skipgram_ns["vocab"]= idx2word
skipgram_ns["stoi"]= word2idx


with open("skipgram_ns.pkl", "wb") as f:
    pickle.dump(skipgram_ns, f)

#files.download("skipgram_ns.pkl")
outputs = produce_neighbor_visualizations(skipgram_ns["embeddings"], skipgram_ns["stoi"], skipgram_ns["vocab"],"skipgram_ns",sample_k=50, top_k=10, seed=42)
outputs["tsne"].write_html("skipgram_ns_tsne.html")
outputs["umap"].write_html("skipgram_ns_umap.html")
#files.download("skipgram_ns_tsne.html")
#files.download("skipgram_ns_umap.html")


def generate_cbow_pairs(sentences, window_size):
    pairs = []
    for sentence in sentences:
        length = len(sentence)
        for center_pos, center_word in enumerate(sentence):
            start = max(0, center_pos - window_size)
            end = min(length, center_pos + window_size + 1)

            context = [sentence[i] for i in range(start, end) if i != center_pos]
            if context:
              pairs.append((context, center_word))
    return pairs


'''
Implementation of cbow architecture with negative sampling
'''
class cbow_with_neg_sampling:
    def __init__(self, vocab_size, embed_dim=100, learning_rate=0.05, neg_count=5, table_size=int(1e7)):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = learning_rate
        self.neg_count = neg_count

        self.W_in = np.random.uniform(-0.5/self.embed_dim, 0.5/self.embed_dim, (vocab_size, embed_dim)).astype(np.float32)
        self.W_out = np.random.uniform(-0.5/self.embed_dim, 0.5/self.embed_dim, (vocab_size, embed_dim)).astype(np.float32)

        self.unigram_table = None
        self.table_size = 0
        self.name="cbow_ns"

    def build_unigram_table(self, counts):
        scaled_counts = counts ** 0.75
        repeats = np.round(scaled_counts).astype(int)
        table = np.repeat(np.arange(len(counts)), repeats)
        self.table_size = len(table)
        return table.astype(np.int32)

    def build_unigram(self, counts):
        self.unigram_table = self.build_unigram_table(counts)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, pairs, epochs=5, batch_size=1024, a=None, b=None):
        for epoch in range(epochs):
            random.seed(42 + epoch)
            random.shuffle(pairs)

            batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
            total_loss = 0
            for batch in batches:
                context_lists, centers = zip(*batch)
                loss = self.train_batch(context_lists, centers)
                total_loss += loss * len(centers)

            print(f"Epoch {epoch+1}, Loss={total_loss/len(pairs):.4f}")

    def train_batch(self, context_lists, centers):
        batch_size = len(centers)
        centers = np.array(centers, dtype=np.int32)

        h_list = []
        for context in context_lists:
            context_words_input_embeddings = self.W_in[np.array(context)]
            h=np.mean(context_words_input_embeddings, axis=0)
            h_list.append(h)
        h = np.vstack(h_list)


        U_o = self.W_out[centers]
        scores_pos = np.sum(h * U_o, axis=1)
        p_pos = self.sigmoid(scores_pos)
        loss = -np.sum(np.log(p_pos + 1e-100))

        grad_pos = self.lr * (1 - p_pos)[:, None]
        self.W_out[centers] += grad_pos * h
        grad_h = grad_pos * U_o


        neg_samples = np.random.randint(self.table_size, size=(batch_size, self.neg_count))
        neg_samples = self.unigram_table[neg_samples]

        U_neg = self.W_out[neg_samples]
        scores_neg = np.einsum('ij,ikj->ik', h, U_neg)
        p_neg = 1 / (1 + np.exp(scores_neg))
        loss -= np.sum(np.log(p_neg + 1e-100))

        grad_neg = self.lr * (1 - p_neg)
        self.W_out[neg_samples] -= grad_neg[:, :, None] * h[:, None, :]
        grad_h -= np.sum(grad_neg[:, :, None] * U_neg, axis=1)
        for i, ctx in enumerate(context_lists):
            for w in ctx:
                self.W_in[w] += grad_h[i] / len(ctx)

        return loss / batch_size

flat_tokens = [w for sent in data for w in sent]
word2idx, idx2word, counts = build_vocab(flat_tokens,5)

encoded_sentences = [[word2idx[w] for w in sent if w in word2idx] for sent in data]
pairs = generate_cbow_pairs(encoded_sentences, window_size=2)
cbow_ns_ = cbow_with_neg_sampling(len(word2idx), embed_dim=100, learning_rate=0.01, neg_count=5)
cbow_ns_.build_unigram(counts)
cbow_ns_.train(pairs, epochs=30, batch_size=32,a=word2idx,b=idx2word)

cbow_ns={}
cbow_ns["embeddings"]= cbow_ns_.W_in
cbow_ns["vocab"]= idx2word
cbow_ns["stoi"]= word2idx

with open("cbow_ns.pkl", "wb") as f:
    pickle.dump(cbow_ns, f)

#files.download("cbow_ns.pkl")
outputs = produce_neighbor_visualizations(cbow_ns["embeddings"], cbow_ns["stoi"], cbow_ns["vocab"],"cbow_ns",sample_k=50, top_k=10, seed=42)
outputs["tsne"].write_html("cbow_ns_tsne.html")
outputs["umap"].write_html("cbow_ns_umap.html")
#files.download("cbow_ns_tsne.html")
#files.download("cbow_ns_umap.html")
