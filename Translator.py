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