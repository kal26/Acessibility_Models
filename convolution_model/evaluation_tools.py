import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Must be before importing keras!
import sys
sys.path.append('/home/kal/TF_models/bin/')
import tf_memory_limit
#import general use packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ucscgenome
from tqdm import tqdm
from itertools import zip_longest, product, chain, repeat

#import keras related packages
from keras import backend as K
from keras.models import load_model, Model, Input
from keras.layers import Input, Lambda
import tensorflow as tf
#import custom packages
import helper
import viz_sequence
import atacseq

#some batch forming methods
def blank_batch(seq, batch_size=32):
     """Make a batch blank but for the given sequence in position 0."""
     batch = np.zeros((batch_size, seq.shape[0], seq.shape[1]), dtype=np.uint8)
     batch[0] = seq
     return batch

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def filled_batch(iterable, batch_size=32, fillvalue=np.asarray([False]*256*4).reshape(256,4)):
    """Make batches of the given size until running out of elements, then buffer."""
    groups = grouper(iterable, batch_size, fillvalue=fillvalue)
    while True:
        yield np.asarray(next(groups))

def random_seq():
    """return a random Sequence."""
    random = np.random.choice(np.fromstring('acgt', np.uint8), size=256)
    return atacseq.Sequence(random)

def sample_gumbel(shape, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [sequence length, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

class ATACmodel(object):
    """Acessibility keras model."""

    def __init__(self, full_path, output_path=None, model_path=None):
        """Create a new model object.

        Arguments:
            model_path -- path to a trained mode's directory with
                          final_model.hdf5
                          32.3_32.3_16.3_8.3_model.png
                          history/
                          intermediate_weights/
                          atac_analysis/
                          evaluation/
        Keywords:
            output_path -- directory to write out requested files to (defalut is to evaluation or atac analysis). 
            model_path -- actual path to the model, default is 'final_model.hdf5'.
        """
        self.full_path = full_path
        if model_path == None:
            self.model_path = os.path.join(self.full_path, 'final_model.h5')
        else:
            self.model_path = model_path
        self.model = load_model(self.model_path)
        if output_path != None:
            self.out_path = output_path
        else:
            self.out_path = os.path.join(full_path, 'evaluation')
        self.layer_dict = dict([(layer.name, layer) for layer in self.model.layers]) 
        self.get_act = K.function([self.model.input, K.learning_phase()], [self.model.output])

    def __str__(self):
        """Printable version of the model."""
        return self.model.summary()
    
    def __repr__(self):
        """Representaiton of the model."""
        return 'TFmodel() at ' + self.model_path

    def get_activation(self, generator, distribution_repeats=32):
        """Return predctions for the given sequence or generator.
          
        Arguments:
            generator -- A sequence or generator of correct length sequences.
        Keywords:
            distribution_repeats -- Number of sequences to sample from a distribution.
        Output:
            outact -- List or value for activation.
        """
        activation = list()
        #assume iterable
        try:
            test = next(generator)
            if isinstance(test, atacseq.ATACDist):
                dist = True
                #distribution
                #converts distributions to discrete sequences and averages
                def gen():
                    for i in range(distribution_repeats):
                        yield test.discrete_seq()
                    for elem in generator:
                        for i in range(distribution_repeats):
                            yield elem.discrete_seq()
                g = gen()
                batch_gen = filled_batch(g)
            else:
                dist = False
                def stackgen():
                    yield test
                    for elem in generator:
                        yield elem
                g = stackgen()
                batch_gen = filled_batch(g)
            # get the numbers
            for batch in batch_gen:
                activation.append(self.get_act([batch, 0]))
            activation = np.asarray(activation).flatten()
            if dist:
                #average every so often
                ids = np.arange(len(activation))//distribution_repeats
                outact = np.bincount(ids, activations)/np.bincount(ids) 
                return outact
            return activation
        except TypeError:
            #acutally not iterable
            if isinstance(generator, atacseq.ATACDist):
                dist = True
                #distribution
                #converts distributions to discrete sequences and averages
                def gen():
                    for i in range(distribution_repeats):
                        yield generator.discrete_seq()
                g = gen()
                batch_gen = filled_batch(g)
                for batch in batch_gen:
                    activation.append(self.get_act([batch, 0]))
                activation = np.asarray(activation).flatten()
                return np.sum(activation)/activation.shape[0]
            else:
                return self.get_act([blank_batch(generator.model_input()), 0])[0][0][0] 

    def get_importance(self, seq, viz=False, start=None, end=None, plot=False, temp=.1):
        """Generate the gradient based importance of a sequence according to a given model.
        
        Arguments:
             seq -- the Sequence to run through the keras model.
             viz -- sequence logo of importance?
             start -- plot only past this nucleotide.
             end -- plot only to this nucleotide.
             plot -- generate a gain-loss plot?
        Outputs:
             diffs -- difference at each position to score.
             average_diffs -- base by base importance value. 
             masked_diffs -- importance for bases in origonal sequence.
        """
        score = self.get_activation(seq)
        mutant_preds = self.get_activation(seq.ngram_mutant_gen())
        #get the right shape
        mutant_preds = mutant_preds.reshape((-1, 4))[:len(seq.seq)]
        diffs = mutant_preds - score
        # we want the difference for each nucleotide at a position minus the average difference at that position
        average_diffs = list()
        for base_seq, base_preds in zip(seq.seq, mutant_preds):
            this_base = list()
            for idx in range(4):
                this_base.append(base_preds[idx] - np.average(base_preds))
            average_diffs.append(list(this_base))
        average_diffs = np.asarray(average_diffs)
        # masked by the actual base
        masked_diffs = (seq.seq * average_diffs)
        if plot:
            # plot the gain-loss curve 
            plt.figure(figsize=(20, 2))
            plt.plot(np.amax(diffs, axis=1)[start:end])
            plt.plot(np.amin(diffs, axis=1)[start:end])
            plt.title('Prediciton Difference for a Mutagenisis Scan')
            plt.ylabel('importance (difference)')
            plt.xlabel('nucleotide')
            plt.show()
        if viz:
            temp = temp
            #print('Prediciton Difference')
            #viz_sequence.plot_weights(average_diffs[start:end])
            print('Masked average prediciton difference')
            viz_sequence.plot_weights(masked_diffs[start:end])
            #print('Softmax prediction difference')
            #viz_sequence.plot_weights(helper.softmax(diffs[start:end]))
            print('Information Content of Softmax prediction difference')
            viz_sequence.plot_icweights(helper.softmax(diffs[start:end]/(temp*self.get_activation(seq))))
        return diffs, average_diffs, masked_diffs


    def gumbel_dream(self, seq, dream_type, temp=10, layer_name='final_output', filter_index=0, meme_library=None, num_iterations=20, step=None, constraint=None, viz=False):
        """ Dream a sequence for the given number of steps employing the gumbel-softmax reparamterization trick.

        Arguments:
            seq -- SeqDist object to iterate over.
            dream_type -- type of dreaming to do. 
                standard: update is average gradient * step
                constrained: dream the rejection of this model against the other model.
        Keywords:
            temp -- for gumbel softmax.
            layer_name -- name of the layer to optimize.
            filter_index -- which of the neurons at this filter to optimize.
            meme_library -- memes to use if applicable (default is CTCF)
            num_iterations -- how many iterations to increment over.
            step -- default is 1/10th the initial maximum gradient
            constraint -- for constrained dreaming, the model to use for rejection.
            viz -- sequence logo of importance?
        Returns:
            dream_seq -- result of the iterations.
        """
        # dreaming won't work off of true zero probabilities - if these exist we must add a pseudocount
        if np.count_nonzero(seq.seq) != np.size(seq.seq):
            print('Discrete Sequence passed - converting to a distibution via pseudocount')
            dream_seq = atacseq.ATACDist(helper.softmax(3*seq.seq + 1), atac_counts=seq.atac_counts)
        else:
            dream_seq = atacseq.ATACDist(seq.seq, atac_counts=seq.atac_counts)

        # get a gradient grabbing op
        #input underlying distribution as (batch_size, 1024, 4) duplications of the sequence
        dist = tf.placeholder(shape=((1024,4)), name='distribution', dtype=tf.float32)
        logits_dist = tf.reshape(dist, [-1,4])
        # sample and reshape back (shape=(batch_size, 1024, 4))
        # set hard=True for ST Gumbel-Softmax
        sampled_seq = tf.reshape(gumbel_softmax(logits_dist, temp, hard=True),[-1, 1024, 4])
        sampled_input = tf.concat([seq.atac_counts, sampled_seq], 1) 
        sampled_input = self.model.input
        if layer_name == 'final_output':
            loss = self.model.output
        else:
            max_by_direction = Lambda(lambda x: K.maximum(K.max(x[:x.shape[0]//2, :, :], axis=1), K.max(x[x.shape[0]//2:, ::-1, :], axis=1)), name='stackmax', output_shape=lambda s: (s[0] // 2, 1))
            layer_output = max_by_direction(self.layer_dict[layer_name].output)
            loss = layer_output[:, filter_index] #each batch and nuceotide at this neuron.
        # compute the gradient of the input seq wrt this loss and average to get the update (sampeling already weights for probability)
        if dream_type == 'constrained':
             sampled_seq = constraint.model.input
             pwm_loss = constraint.output
             grads = K.gradients(loss, sampled_seq)[0]
             pwms = K.gradients(pwm_loss, sampled_seq)[0] 
             update = K.mean(helper.rejection(grads, pwms), axis=0)
        else:
            update = K.mean(K.gradients(loss, sampled_seq)[0], axis=0)
        #get a function
        update_op = K.function([sampled_input, K.learning_phase()], [update])

        #find a step size
        if step == None:
            step = 1/(np.amax(update_op([[dream_seq.seq]*32, 0])[0]))
            print('Step ' + str(step))
        # print the initial sequence
        if viz:
            print('Initial Sequence')
            seq.logo()
            print('Model Prediction: ' + str(self.model.predict(blank_batch(dream_seq.discrete_seq()))[0][0]))
            self.get_importance(dream_seq, viz=True)
            print('PWM score: ' + str(dream_seq.find_pwm(viz=True)[2]))

        #iterate and dream
        for i in range(num_iterations):
            update = update_op([[dream_seq.seq]*32, 0])[0]
            if dream_type == 'standard':
                dream_seq.seq = helper.softmax(np.log(dream_seq.seq) + update*step)
            elif dream_type == 'adverse':
                dream_seq.seq = helper.softmax(np.log(dream_seq.seq) + update*step -1) 
            elif dream_type == 'blocked':
                meme, position, _ = dream_seq.find_pwm(meme_library=meme_library)
                update[position:position+meme.seq.shape[0]] = 0
                dream_seq.seq = helper.softmax(np.log(dream_seq.seq) + update*step)
            if i%(num_iterations//4) == 0 and viz:
                print('Sequence after ' + str(i) + ' iterations')
                viz_sequence.plot_icweights(dream_seq.seq)

        #print the final sequence
        if viz:
            print('Final sequence')
            dream_seq.logo()
            print('Model Prediction: ' + str(self.model.predict(blank_batch(dream_seq.discrete_seq()))[0][0]))
            self.get_importance(dream_seq, viz=True)
            print('PWM score: ' + str(dream_seq.find_pwm(viz=True)[2]))
        return dream_seq     

    def build_iterate(self, layer_name='final_output', filter_index=0):
        """ Build a interation operation for use with dreaming method.
     
        Keywords:
           layer_name -- layer dictionary enry to get the output from.
           filter_index -- inex of the filter to pull from the layer. 
        Output:
            iterate_op -- iteration operation returning gradients.
        """
        # set a placeholder input
        encoded_seq = self.model.input
        # build a function that sumes the activation of the nth filter of the layer considered
        if layer_name == 'final_output':
            activations = self.model.output
            # compute the gradient of the input picture wrt this loss
            grads = K.gradients(K.mean(activations), encoded_seq)[0]
        else:
            layer_output = self.layer_dict[layer_name].output
            activations = layer_output[:, :, filter_index] #each batch and nuceotide at this neuron.
            # forward and reverse sequences
            combined_activation = K.mean(np.maximum(activations[:32], activations[32:]))
            # compute the gradient of the input picture wrt this loss
            grads = K.gradients(combined_activation, encoded_seq)[0]
            # normalization trick: we normalize the gradient - not sure if I should use this
            # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        # this function returns the loss and grads given the input picture
        iterate_op = K.function([encoded_seq, K.learning_phase()], [grads])
        return iterate_op 


def snv_gen(peaks, genome, alt=False):
    """Generate sequnces from snv data.
    
    Arguments:
        peaks -- from a bed file.
        genome -- to pull bed from.
    Keywords:
        alt -- give alternate allele version.
    Returns:
        seq -- sequence with the alternate or refernce allele, centered around the position. """
    for index, row in peaks.iterrows():
        if row.position > 128:
            seq = atac_seq.encode_to_onehot(genome[row.chr][row.position-128:row.position+128])
            if alt:
                seq[128] = atacseq.encode_to_onehot(row.altAllele.lower())
            else:
                seq[128] = atacseq.encode_to_onehot(row.refAllele.lower())
        else:
            # sequence too close to begining
            seq = atacseq.encode_to_onehot(genome[row.chr][0:256])
            if alt:
                seq[row.position] = atacseq.encode_to_onehot(row.altAllele.lower())
            else:
                seq[row.position] = atacseq.encode_to_onehot(row.refAllele.lower())
        if seq.shape != (256, 4):
            # seq too close to end?
            print('Sequence at ' +str(row.chr) + ' ' + str(row.position) + ' is too short!')
            offset = 0
            while seq.shape != (256, 4) and offset < 128:
                offset += 1
                seq = atacseq.encode_to_onehot(genome[row.chr][row.position-128-offset:row.position+128-offset])
            if alt:
                seq[128-offset] = atacseq.encode_to_onehot(row.altAllele.lower())
            else:
                seq[128-offset] = atacseq.encode_to_onehot(row.refAllele.lower())
        if (row.refAllele).lower() != (genome[row.chr][row.position]).lower():
             raise IndexError('Reference allele does not match reference genome')
        if seq.shape == (256, 4):
            yield seq
        else:
            print('Sequence at ' +str(row.chr) + ' ' + str(row.position) + ' couldn\'t be fixed')


def group_stats(key, h1, h2, h3):
    # Summarize history for accuracy
    out1 = np.copy(h1[key])
    out2 = np.copy(h2[key])
    out3 = np.copy(h3[key])
    return np.concatenate([out1, out2, out3])

