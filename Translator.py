# this project will use a pretrained model and then have GluonNLP evaluate the said model
# and will allow for English to German translations
# the project will be very useful to me and my classmates because we realized we did not learn
# enough german before studying in Germany :)
# this project utilizes and builds on examples from the GluonNLP API which is linked below
# https://nlp.gluon.ai

import warnings
warnings.filterwarnings('ignore')
# disabled the warnings because there were too many

import random
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
nlp.utils.check_version('0.7.0')

# setting up the enviroment with the Gluon API recommended settings 

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.gpu(0) 

import nmt

wmt_model_name = 'transformer_en_de_512'

wmt_transformer_model, wmt_src_vocab, wmt_tgt_vocab = \
    nlp.model.get_model(wmt_model_name, dataset_name='WMT2014',pretrained=True,ctx=ctx)

# we are using a model that has a mixed vocab of EN-DE, so the source and target language vocab are the same
print(len(wmt_src_vocab), len(wmt_tgt_vocab))


# processing steps 
# clip the source and target sequences via nlp.data.WMT2014BPE
# split the string input to a list of tokens
# map the string token into its index in the vocabulary
# append EOS token to source sentence and add BOS and EOS tokens to target sentence.
import hyperparameters as hparams


wmt_data_test = nlp.data.WMT2014BPE('newstest2014', src_lang=hparams.src_lang, tgt_lang=hparams.tgt_lang)
print('Source language %s, Target language %s' % (hparams.src_lang, hparams.tgt_lang))
print('Sample BPE tokens: "{}"'.format(wmt_data_test[0]))

wmt_test_text = nlp.data.WMT2014('newstest2014',src_lang=hparams.src_lang, tgt_lang=hparams.tgt_lang)
print('Sample raw text: "{}"'.format(wmt_test_text[0]))

wmt_test_tgt_sentences = wmt_test_text.transform(lambda src, tgt: tgt)
print('Sample target sentence: "{}"'.format(wmt_test_tgt_sentences[0]))



import dataprocessor

print(dataprocessor.TrainValDataTransform.__doc__)

# wmt_transform_fn includes the four preprocessing steps mentioned above.
wmt_transform_fn = dataprocessor.TrainValDataTransform(wmt_src_vocab, wmt_tgt_vocab)
wmt_dataset_processed = wmt_data_test.transform(wmt_transform_fn, lazy=False)
print(*wmt_dataset_processed[0], sep='\n')

def get_length_index_fn():
    global idx
    idx = 0
    def transform(src, tgt):
        global idx
        result = (src, tgt, len(src), len(tgt), idx)
        idx += 1
        return result
    return transform

wmt_data_test_with_len = wmt_dataset_processed.transform(get_length_index_fn(), lazy=False)
