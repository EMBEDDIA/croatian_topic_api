# import csv
import os
# import logging
# import pickle
import torch
import pickle
import string
# from gensim.models import KeyedVectors
# from . import data

import pandas as pd
import numpy as np
from collections import Counter


import torch.nn.functional as F
import numpy as np
import math

from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## define model and optimizer
# model = ETM(100, vocab_size, args.t_hidden_size, args.rho_size, args.emb_size,
#                 args.theta_act, embeddings, args.train_embeddings, args.enc_drop).to(device)

class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size, t_hidden_size, rho_size, emsize,
                 theta_act, embeddings=None, train_embeddings=True, enc_drop=0.5):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.emsize = emsize
        self.t_drop = nn.Dropout(enc_drop)

        self.theta_act = self.get_activation(theta_act)

        ## define the word embedding matrix \rho
        if train_embeddings:
            self.rho = nn.Linear(rho_size, vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = nn.Embedding(num_embeddings, emsize)
            self.rho = embeddings.clone().float().to(device)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)  # nn.Parameter(torch.randn(rho_size, num_topics))

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, t_hidden_size),
            self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        try:
            logit = self.alphas(self.rho.weight)  # torch.mm(self.rho, self.alphas)
        except:
            logit = self.alphas(self.rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0)  ## softmax over vocab dimension
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, kld_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def forward(self, bows, normalized_bows, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * bows).sum(1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta


class ModelLoad():

    def __init__(self):
        # configuration
        self.ROOT_FOLDER = os.path.dirname(__file__)

        print('ROOT_FOLDER',self.ROOT_FOLDER)


        # Load a trained topic model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        DIRECTORIES = {
            'ml_topic_model_path': os.path.join(self.ROOT_FOLDER, 'models/ml_topic_model')
        }
        self.model_file = os.path.join(DIRECTORIES['ml_topic_model_path'], 'topic_model.pt')
        # self.word2vec_file = os.path.join(DIRECTORIES['ml_topic_model_path'], 'word2vec')
        self.etm_vocab_file = os.path.join(DIRECTORIES['ml_topic_model_path'], 'topic_vocab.pkl')

        self.topic_label_file = os.path.join(DIRECTORIES['ml_topic_model_path'], 'croatian_topic_labels.csv')

        print('model_file',self.model_file)
        print('model_dir',os.listdir(os.path.join(self.ROOT_FOLDER, 'models')))
        print('model_dir_s',os.listdir(DIRECTORIES['ml_topic_model_path']))
        print(os.path.isfile(self.model_file))
        if not os.path.isfile(self.model_file):
            print('Please Download the model ...')
            exit(0)

        print("Loading trained ETM from", self.model_file)
        # Load trained ETM vocab
        self.vocab = pickle.load(open(self.etm_vocab_file, "rb"))
        print("Vocab of trained ETM:", len(self.vocab))

        # Dummy Embedding
        # TODO: this is really stupid way to do this
        embeddings = np.zeros((len(self.vocab), 300))
        embeddings = torch.from_numpy(embeddings).to(device)
        # Load
        self.model = ETM(num_topics=100,vocab_size=len(self.vocab),t_hidden_size=800,rho_size=300,emsize=300,theta_act='relu',train_embeddings=0,embeddings=embeddings)
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()
        self.model.to(self.device)

        #Load Topic Lables
        self.topic_label = pd.read_csv(self.topic_label_file)



    def load_models(self):
        #Return Model and word2vec
        return self.model

    def get_model(self):
        # Return Model
        return self.model

    def get_vocab(self):
        # Return Vocabulary
        return self.vocab

    def get_topic_labels(self):
        # Return Topic Labels
        return self.topic_label

    def get_device(self):
        return self.device


def get_batch(tokens, counts, ind, vocab_size, device, emsize=300):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))
    print(tokens, ind)
    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        L = count.shape[1]
        if len(doc) == 1:
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch


def infer_doc_topics(model, vocab, documents, min_prob=0.01):
    # convert documents to BoW represenation
    tokens = []
    counts = []
    non_empty_docs = []
    if isinstance(documents,str):
        documents = [documents]
    for d, doc in enumerate(documents):
        doc = doc.lower().split()
        tokens_doc = sorted([vocab.index(w) for w in doc if w.lower() in vocab])
        if len(tokens_doc) > 0:
            word_counts = Counter(doc)
            counts_doc = [word_counts[vocab[tok_index]] for tok_index in tokens_doc]
            tokens.append(np.array([tokens_doc]))
            counts.append(np.array([counts_doc]))
            non_empty_docs.append(doc)
    if len(non_empty_docs) == 0:
        return []
    tokens = np.array(tokens)
    counts = np.array(counts)
    indexes = list(range(len(tokens)))
    vocab_size = len(vocab)
    data_batch = get_batch(tokens, counts, indexes, vocab_size, device)
    sums = data_batch.sum(1).unsqueeze(1)
    normalized_data_batch = data_batch / sums
    theta, _ = model.get_theta(normalized_data_batch)
    if device != 'cpu':
        theta = theta.cpu().detach().numpy()
    else:
        theta = theta.detach().numpy()
    theta[theta < min_prob] = 0.0
    return theta

def predict_ml_tp(data, model, vocab,topic_labels, topk=5):
    min_prob = 0.01
    topics = {}
    topics['suggested_label'] = []
    topics['description'] = []
    topics['topic_words'] = []

    for val in data:
        topic_vec = infer_doc_topics(model, vocab, [val], min_prob=min_prob)
        if len(topic_vec) == 0:
            topic_vec = np.zeros(model.num_topics)
        else:
            topic_vec = topic_vec[0] / topic_vec[0].sum()

        arr = -1 * topic_vec
        indices = np.argpartition(arr, topk - 1)[:topk]
        topic_vec_str = topic_labels.iloc[indices, :].values

        suggested_label =[]
        description = []
        topic_words = []
        for i, tmp in enumerate(topic_vec_str):
            key = 'topic_top' + str(i)
            suggested_label.append(key + " : "+tmp[0])
            description.append(key + " : "+tmp[1])
            topic_words.append(key + " : "+tmp[2])
        topics['suggested_label'].append(suggested_label)
        topics['description'].append(description)
        topics['topic_words'].append(topic_words)

    return topics



model_load = None


def predict(data):
    global model_load
    if model_load is None:
        model_load = ModelLoad()
        # print(model_load.get_model().logsigma_q_theta.weight)
    topics = predict_ml_tp(data, model_load.get_model(),model_load.get_vocab(), model_load.get_topic_labels())
    # print(topics)
    return topics

# test=['Želiš pronaći ženu za jednu noć? Dobrodošli na']
# predict(test)
