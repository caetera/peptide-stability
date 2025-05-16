############# Library Import 
import re
import tensorflow
import pandas as pd
import streamlit as st
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.initializers import Constant
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.utils import compute_class_weight

#constants
aa_dict = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
alphabet = ''.join(aa_dict.keys())
pool_length = 2

#functions
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

def seqValidator(seq):
    if len(seq) >=7 and len(seq) <=30:
        invalid = re.findall(f'[^{alphabet}]', seq)
        if len(invalid) == 0:
            return ''
        else:
            return 'Unknown symbols in sequence: ' + ' '.join([f'"{c}"' for c in set(invalid)])
    else:
        return 'Sequence length should be from 7 to 30 AA'

def translate_sequence(str1):
        a = []
        for i in range(len(str1)):
            a.append(aa_dict.get(str1[i]))
        return a
        
def translate_result(prob):
    if prob > 0.5:
        return ('unstable', 100 * prob)
    else:
        return ('stable', 100 * (1 - prob))

# load Model For Gender Prediction
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
 
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
 
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
 
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
 
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def get_config(self):
      config = super().get_config().copy()
      config.update({
          
          'W_regularizer' : self.W_regularizer,
          'u_regularizer' : self.u_regularizer,
          'b_regularizer' : self.b_regularizer,
 
          'W_constraint' : self.W_constraint,
          'u_constraint' : self.u_constraint,
          'b_constraint' : self.b_constraint,
 
          'bias' : self.bias})
      return config    
 
    def build(self, input_shape):
        assert len(input_shape) == 3
 
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
 
        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
 
        super(AttentionWithContext, self).build(input_shape)
 
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
 
        a = K.exp(ait)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

 
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
 
        if self.bias:
            uit += self.b
 
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
 
        a = K.exp(ait)
 
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
 
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)
 
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


model = load_model('model.h5', custom_objects={'AttentionWithContext' : AttentionWithContext})

st.set_page_config(page_title='Peptide Stability prediction')
st.title("Peptide Stability prediction")

st.sidebar.subheader(("Abstract"))
st.sidebar.markdown("In proteomics peptides are used as surrogates for protein quantification and therefore peptides selection is crucial for good protein quantification.At the same time large-scare proteomics studies involve hundreds of samples and take several weeks of measurement time. Thefore, the study of peptide stability during the duration of the project is essencial for quantification results with high accuracy and precision. The goal of this webserver is to predict the stability of peptides.")

st.sidebar.subheader(("Please Read requirements"))

st.sidebar.markdown(f"- Valid amino acid letters ({alphabet})")
st.sidebar.markdown("- Valid peptide length is from 7 to 30 amino acids")
st.sidebar.markdown("- No Post-translational modifications are supported")

caption= "The proposed methodology to develop Peptide/Protein Stability classifier"

st.subheader(("Input Sequence(s)"))
seq_string = st.text_area("Ex: LAENVKIK", height=200)

if st.button("PREDICT"):
    if (seq_string==""):
        st.error("Please input the sequence first")
        exit()

    #lets try to guess the delimeter
    for delimeter in [';', ',', '\n']:
        if seq_string.find(delimeter) != -1:
             break
    
    #split by the delimeter and strip sequences
    sequences = pd.DataFrame([s.strip() for s in seq_string.split(delimeter)], columns=['sequence'])
    #validation of proper protein string
    sequences['error'] = sequences['sequence'].apply(seqValidator)

    valid_sequences = sequences.loc[sequences['error'] == '', ['sequence']].reset_index(drop=True)
    
    if (valid_sequences.shape[0] > 0):
        valid_sequences['encoding'] = valid_sequences['sequence'].apply(translate_sequence)

        Sequence = tensorflow.keras.preprocessing.sequence.pad_sequences(valid_sequences['encoding'], maxlen=50, padding='post')
        valid_sequences['prediction'] = model.predict(Sequence)
        valid_sequences[['stability', 'probability (%)']] = pd.DataFrame(valid_sequences['prediction'].apply(translate_result).tolist())

        valid_sequences.drop(['encoding', 'prediction'], axis=1, inplace=True)

        st.subheader('Predicted peptides')
        st.dataframe(valid_sequences, hide_index=True, column_config={'probability (%)': st.column_config.NumberColumn(format='%.2f')})
        st.download_button('Download result table',
                           valid_sequences.to_csv(index=False).encode('utf-8'),
                           'prediction.csv',
                           'text/csv')
    
    else:
        st.subheader('No valid input')

    if sum(sequences['error'] != '') > 0:
        st.subheader('Not-predicted peptides')
        st.table(sequences.loc[sequences['error'] != '', :])

