import numpy as np
import pandas as pd
import keras.preprocessing.text as Text


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


class Config():
    image_path = '/data3/flickr30k/feats.npy'
    annotation_path = '/data3/flickr30k/results_20130124.token'


config = Config

images, caption = data_setting(config.image_path, config.annotation_path)
# images / caption : ndarray

word2idx, idx2word, bias_init_vector = preprocessingVoca(caption)





#index = range(caption.shape[0]) # list / size = 158915
#np.random.shuffle(index)
#images = images[index]
#caption = caption[index]
