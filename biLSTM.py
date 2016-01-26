import numpy as np
import pandas as pd
import tensorflow as tf
import keras.preprocessing.sequence as Sequence
from tensorflow.models.rnn import rnn_cell


def data_setting(_image_path,_annotation_path):
    _images = np.load(_image_path)
    _annotation = pd.read_table(_annotation_path,'\t',header=None,names=['image','caption'])
    _annotation = _annotation['caption'].values # ndarray
    return _images, _annotation

def preprocessingVoca(_caption):

    _caption = map(lambda x: x.lower().split(' '), _caption)  # list / len = 159815
    _vocas = {}

    for i in range(len(_caption)):
        for _voca in _caption[i]:
            _vocas[_voca] = _vocas.get(_voca, 0) + 1

    _order = 1
    _idx2word = {}
    _vocas['.']=0
    _idx2word[0] = '.'

    for k in range(len(_vocas.values())):
        if _vocas.values()[k]>=30:
            _idx2word[_order] = _vocas.keys()[k]
            _order += 1

    _word2idx = dict(zip(_idx2word.values(), _idx2word.keys()))
    _word2idx['#START#']=0
    _word2idx['.']= _order

    _bias_init_vector = np.array([1.0*_vocas[_idx2word[p]] for p in _idx2word])
    _bias_init_vector /= np.sum(_bias_init_vector) # normalize to freq
    _bias_init_vector = np.log(_bias_init_vector)
    _bias_init_vector -= np.max(_bias_init_vector) # shift to nice numeric range

    return _word2idx, _idx2word, _bias_init_vector


def preprocessingCaption(_cap, wordtoidx):

    _cap = map(lambda cap: [wordtoidx[word] for word in cap.lower().split(' ')[:-1] if word in wordtoidx], _cap)
    max_steps = np.max(map(lambda x: len(str(x).split(' ')),_cap)) # 79
    _cap = Sequence.pad_sequences(_cap, maxlen=max_steps+1, padding='post') # ndarray
    _cap = Sequence.pad_sequences(_cap, maxlen=max_steps+2, padding='pre')

    return _cap, max_steps


def biLSTM(_x,_y,_mask,_weights,_biases,_config):

    _x = tf.matmul(_x, _weights['img_emb']) + _biases['img_emb']

    lstm = rnn_cell.BasicLSTMCell(_config.hidden)
    state_lstm = tf.zeros([_config.batch_size, lstm.state_size])
    output_lstm, state_lstm = lstm(_x, state_lstm) # output : h // state : concat c & h






    return output_lstm


class Config():
    image_path = '/data3/flickr30k/feats.npy'
    annotation_path = '/data3/flickr30k/results_20130124.token'
    img_size = 4096
    emb_size = 256
    hidden = 256
    batch_size = 128
    epoch = 1000





config = Config
images, caption = data_setting(config.image_path,
                               config.annotation_path) # images / caption : ndarray


word2idx, idx2word, bias_init_vector = preprocessingVoca(caption)


index = range(caption.shape[0]) # list / size = 158915
np.random.shuffle(index)
images = images[index]
caption = caption[index]


config.n_words = len(word2idx)
cap, config.max_steps = preprocessingCaption(caption,word2idx)


x = tf.placeholder(tf.float32, [config.batch_size, config.img_size])
y = tf.placeholder(tf.int32, [config.batch_size, config.max_steps+2])
mask = tf.placeholder(tf.float32, [config.batch_size, config.max_steps+2])


with tf.device("/cpu:0"):
    Wemb = tf.Variable(tf.random_normal([config.n_words, config.emb_size]))

weights = {'img_emb': tf.Variable(tf.random_normal([config.img_size,config.emb_size])),
           'out': tf.Variable(tf.random_normal([config.emb_size,config.n_words]))}
biases = {'img_emb': tf.Variable(tf.random_normal([config.emb_size])),
          'out': tf.Variable(tf.random_normal([config.n_words])),
          'Wemb': tf.Variable(tf.random_normal([config.emb_size]))}


loss = biLSTM(x, y, mask, weights, biases, config)

