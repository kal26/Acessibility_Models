import sys
import numpy as np
from itertools import zip_longest, product, chain, repeat
import viz_sequence
import train_TFmodel
import helper
from scipy.signal import correlate2d
from colour import Color
import matplotlib.pyplot as plt

one_hot_encoder = np.fromstring('acgt', np.uint8)

# some sequence encoding methods
def encode_to_string(seq):
    "return a string from string, uint8, or onehot"
    if isinstance(seq, str):
        return seq
    elif isinstance(seq, np.ndarray):
        if seq.dtype == np.uint8:
        #uint8 array
            return seq.tostring().decode('UTF-8')
        else:
        #onehot array
            indicies = np.argmax(seq, axis=1)
            return np.asarray([one_hot_encoder[i] for i in indicies]).tostring().decode('UTF-8')
    else:
        raise TypeError('Sequence is not an accepted type')

def encode_to_uint8(seq):
    "return a uint8 from string, uint8, or onehot"
    if isinstance(seq, str):
        return np.fromstring(seq.lower(), dtype=np.uint8)
    elif isinstance(seq, np.ndarray):
        if seq.dtype == np.uint8:
        #uint8 array
            return seq
        else:
        #onehot array
            indicies = np.argmax(seq, axis=1)
            return np.asarray([one_hot_encoder[i] for i in indicies])
    else:
        raise TypeError('Sequence is not an accepted type')

def encode_to_onehot(seq):
    "return a onehot from string, uint8, or onehot"
    if isinstance(seq, str):
        return np.asarray([np.equal(char, one_hot_encoder) for char in np.fromstring(seq.lower(), dtype=np.uint8)])
    elif isinstance(seq, np.ndarray):
        if seq.dtype == np.uint8:
        #uint8 array
            return np.asarray([np.equal(char, one_hot_encoder) for char in seq])
        else:
        #onehot array
            return seq
    else:
        raise TypeError('Sequence is not an accepted type')

def rc(seq):
    """Takes a seq to its reverse complement of same type."""
    onehot = encode_to_onehot(seq)
    rc = onehot[:, ::-1, ::-1]
    if isinstance(seq, str):
        return encode_to_string(rc)
    elif isinstance(seq, np.ndarray):
        if seq.dtype == np.uint8:
        #uint8 array
            return encode_to_uint8(rc)
        else:
        #onehot array
            return rc
    else:
        raise TypeError('Sequence is not an accepted type')

class Sequence(object):
    """ Encoding and variations on a sequence.

    Attributes:
        seq -- onehot encoding of the sequence.
    """
    
    def __init__(self, nucleotides):
        """ Create a sequence object.
        
        Arguments:
            nucleotides -- Sequence in string, np.uint8, or one-hot form.
        """
        self.seq= encode_to_onehot(nucleotides)   

    def __string__(self):
        """ACTG representation of the sequence."""
        return encode_to_string(self.seq)

    def __repr__(self):
        """Information about the sequence."""
        return 'Sequence() length ' + str(self.seq.shape[0])
 
    def logo(self, start=None, end=None):
        """Plot a sequence logo from start to end."""
        viz_sequence.plot_weights(self.seq[start:end])

    def model_input(self):
        return self.seq

    def sequential_mutant_gen(self):
        """Generate sequences with a blank mutation."""
        for idx in range(self.seq.shape[0]):
            new_seq = np.copy(self.seq)
            new_seq[idx] = np.fromstring('x', np.uint8)
            yield new_seq

    def ngram_mutant_gen(self, n=1, padding='valid'):
        """ Generate ngram mutants trying every possible amino acid combination of a length n in a sequence.

        Keywords:
            n -- width of the motif to mutate.
            padding -- valid or same, similar to keras funcitonality.
        """
        done = False
        if padding != 'valid':
            print('Alternative padding not yet supported')
        while not done:
            for idx in range(len(self.seq)):
                if n//2 <= idx <= len(self.seq) - n//2 - 1:
                    first = idx-n//2
                    last = idx+(n+1)//2 
                    #standard case
                    ngrams = product(one_hot_encoder, repeat=n)
                    for gram in ngrams:
                        new_seq = np.copy(self.seq)
                        new_seq[first:last] = encode_to_onehot(np.asarray(gram))
                        yield new_seq
            done = True

    def double_mutant_gen(self, n=1):
        """Generate every possible double mutant."""
        for mut1_seq in self.ngram_mutant_gen(n=n):
            for mut2_seq in Sequence(mut1_seq).ngram_mutant_gen(n=n):
                yield mut2_seq

    def insertion_mutant_gen(self, n=1):
        """Generate every n length insertion."""
        done = False
        while not done:
            for idx in range(len(self.seq)):
                ngrams = product(one_hot_encoder, repeat=n)
                for gram in ngrams:
                    new_seq = np.insert(self.seq, idx, encode_to_onehot(np.asarray(gram)), axis=0)
                    yield new_seq[:256]
            done = True

    def deletion_mutant_gen(self, n=1):
        """Generate every deletion mutant."""
        done = False
        while not done:
            ngrams = product(one_hot_encoder, repeat=n)
            gram = next(ngrams)
            for start_idx in range(len(self.seq)-n):
                del_idx = range(start_idx, start_idx+n)
                new_seq = np.delete(self.seq, del_idx, axis=0)
                new_seq = np.append(new_seq, encode_to_onehot(np.asarray(gram)), axis=0)
                yield new_seq
            done = True

    def motif_insert_gen(self, motif, mode='same'):
        """Insert a given motif at every position."""
        #have i track the middle of the insertion
        for i in range(self.seq.shape[0]):
            new_seq = self.seq.copy()
            if i-motif.shape[0]//2 < 0: # too early
                 if mode == 'same':
                     new_seq[0:i-motif.shape[0]//2 + motif.shape[0]] = motif[motif.shape[0]//2 - i:]
                     yield new_seq
            elif i-motif.shape[0]//2 + motif.shape[0] > new_seq.shape[0]: # too late
                if mode == 'same':
                    new_seq[i-motif.shape[0]//2:new_seq.shape[0]] = motif[:new_seq.shape[0]-i+motif.shape[0]//2]
                    yield new_seq
            else: # just right
                new_seq[i-motif.shape[0]//2:i-motif.shape[0]//2 + motif.shape[0]] = motif
                yield new_seq

    def find_pwm(self, meme_library=None, viz=False):
        """ Convolute a meme with the sequence.
        
        Keywords:
             meme_library -- list of memes to use.
             viz -- sequence logo of importance?
        Output:
             meme -- SeqDist() of the best matching meme.
             position -- start position of the hit.
             score -- correlation score.
        """
        if meme_library==None:
             meme_library = CTCF_memes
        # find the meme and location of the best match.
        score = -np.inf
        position = 0
        meme = meme_library[0]
        for test_meme in meme_library:
            corr = correlate2d(self.seq, test_meme.pwm, mode='valid')
            if np.nanmax(corr) > score:
                score = np.nanmax(corr)
                position = np.nanargmax(corr)
                meme = test_meme
        if viz:
            print('Weighted log-odds of the Sequence Distribution')
            insert = np.zeros(self.seq.shape)
            insert[position:position+meme.pwm.shape[0]] = meme.pwm
            overlap = insert * self.seq
            viz_sequence.plot_weights(overlap)
        return meme, position, score
 
    def run_pwm(self, meme=None, position=None, viz=False):
        """Get the pwm correlation score with a sequence.

        Keywords:
            meme -- SeqDist() of the best matching meme, or library of memes to test.
            position -- start position of the hit.
            viz -- sequence logo of importance?
        Outputs:
            overlap -- overlap which can be summed for the score.
        """
        if meme==None:
            # we need to find everything
            meme, position, score = self.find_pwm()
        elif position==None or isinstance(meme, list):
            # we have the meme/memelist
            meme, position, score = self.find_pwm(meme_library=meme)
        # just get the score
        insert = np.zeros(self.seq.shape)
        insert[position:position+meme.pwm.shape[0]] = meme.pwm
        overlap = insert * self.seq
        if viz:
            print('Weighted log-odds of the Sequence Distribution')
            viz_sequence.plot_weights(overlap)
        return overlap

class SeqDist(Sequence):
    """A sequence, but as a probability distribution.

    Attributes:
        seq -- probability distribution of bases. 
    """

    def __init__(self, distribution):
        """Create a new sequence distribution object."""
        if isinstance(distribution, np.ndarray) and not (distribution.dtype == np.uint8):
            # right type!
            self.seq = helper.softmax(np.log(distribution)) 
        else:
            raise TypeError('Sequence is not an accepted type')
 
    def __repr__(self):
        """Information about the sequence."""
        return 'SeqDist() length ' + str(self.seq.shape[0])

    def logo(self, start=None, end=None):
        """Plot a sequence logo from start to end."""
        viz_sequence.plot_icweights(self.seq[start:end])

    def discrete_gen(self):
        """Create a generator of discrete sequences."""
        while True: 
            yield self.discrete_seq()

    def discrete_seq(self):
        """Return a discrete sequence samples from the continuous distribuiton."""
        discrete = [np.random.choice(one_hot_encoder, p=base) for base in self.seq]
        return encode_to_onehot(np.asarray(discrete))

class Meme(SeqDist):
    """A position weight matrix.
   
    Attirbutes:
        seq -- frequency representation of the seqeunce.
        pwm -- log-odds representaiton of the motif.    
   """

    def __init__(self, dist, pwm):
        """Create a new Meme object."""
        self.seq = helper.softmax(np.log(dist))
        self.pwm = pwm

    def __repr__(self):
        """Information about the sequence."""
        return 'Meme() length ' + str(self.seq.shape[0])

class ATACSeq(Sequence):
    """ A Sequence and matching atac counts."""

    def __init__(self, nucs, atac_counts=None):
        if atac_counts == None:
            super().__init__(nucs[:, 1:])
            self.atac_counts = nucs[:, 0]
        else:
            super().__init__(nucs)
            self.atac_counts = atac_counts

    def model_input(self):
        return np.insert(self.seq.astype(np.float32), 0, self.atac_counts, axis=1)

    def sequential_mutant_gen(self):
        s = super().sequential_mutant_gen
        for nucs in s:
             yield np.insert(nucs.astype(np.float32), 0, self.atac_counts, axis=1)

    def ngram_mutant_gen(self, n=1, padding='valid'):
        n = super().ngram_mutant_gen(n=n, padding=padding)
        for nucs in n:
             yield np.insert(nucs.astype(np.float32), 0, self.atac_counts, axis=1) 
    
    def double_mutant_gen(self, n=1):
        s = super().double_mutant_gen(n)
        for nucs in s:
             yield np.insert(nucs.astype(np.float32), 0, self.atac_counts, axis=1)

    def logo(self, top=None, bottom=None):
        colors = list(Color("blue").range_to(Color("white"), 50))
        [colors.append(c) for c in (Color("white").range_to(Color("red"), 51))]
        if top == None:
            top = np.amax(self.atac_counts)
        if bottom == None:
            bottom = np.amin(self.atac_counts)
    
        #get hightlights!
        color_weights = [int((x-bottom)/(top-bottom)*100) for x in self.atac_counts]
        highlight=dict()
        for i in range(len(self.atac_counts)):
            w = color_weights[i]
            highlight[(colors[w].rgb[0],colors[w].rgb[1], colors[w].rgb[2], .3)] = [(i, i+1)]
        #plot things out
        viz_sequence.plot_weights(self.seq, highlight=highlight)

    def graph(self):
        plt.figure(figsize=(20, 2))
        plt.title('ATAC counts per base')
        plt.plot(self.atac_counts)
        plt.show()



class ATACDist(SeqDist):
    """ A sequence distribution and matching atac counts."""

    def __init__(self, nucs, atac_counts=None):
        if atac_counts == None:
            super().__init__(nucs[:, 1:])
            self.atac_counts = nucs[:, 0]
        else:
            super().__init__(nucs)
            self.atac_counts = atac_counts

    def discrete_seq(self):
        return np.insert(super().discrete_seq().astype(np.float32), 0, self.atac_counts, axis=1)

    
def process_meme(meme_path, transform=False, verb=False):
    """Extract a meme distribution and process.
   
    Arguments:
        meme_path -- file path to a .meme file.
    Keywords:
        transform -- apply normalization and a log transform or use the pre-generated log-odds matrix.
    Outputs:
        meme_list -- List of SeqDist() meme and reverse complements.
    """
    with open(meme_path, 'r') as infile:
        meme_length = -1
        meme_dists = list()
        meme_lods = list()
        # read for the frequencies
        for line in infile.readlines():
            if 'letter-probability matrix' in line:
                meme_length = int(line.split()[5])
                if verb:
                    print('found meme')
                this_meme_lines = list()
            elif meme_length > 0:
                this_meme_lines.append([float(item.strip()) for item in line.split()])
                meme_length = meme_length - 1
            elif meme_length == 0:
                this_meme = np.asarray(this_meme_lines)
                meme_dists.append(this_meme)
                meme_length = -1
        if meme_length == 0:
            this_meme = np.asarray(this_meme_lines)
            meme_dists.append(this_meme)
            meme_length = -1
        # add rcs of memes
        rcs = list()
        for meme in meme_dists:
            rcs.append(meme[::-1, ::-1])
        meme_dists = meme_dists + rcs
    with open(meme_path, 'r') as infile:
        # read for the pwms
        for line in infile.readlines():
            if 'log-odds matrix' in line:
                meme_length = int(line.split()[5])
                this_meme_lines = list()
            elif meme_length > 0:
                this_meme_lines.append([float(item.strip()) for item in line.split()])
                meme_length = meme_length - 1
            elif meme_length == 0:
                this_meme = np.asarray(this_meme_lines)
                meme_lods.append(this_meme)
                meme_length = -1
        if meme_length == 0:
            this_meme = np.asarray(this_meme_lines)
            meme_lods.append(this_meme)
            meme_length = -1
        # add rcs of memes
        rcs = list()
        for meme in meme_lods:
            rcs.append(meme[::-1, ::-1])
        meme_lods = meme_lods + rcs
        if len(meme_lods) == 0:
            #transofrm the memes
            if verb:
                print('using manual log-odds calculation')
            psuedocount=0.005
            for meme in meme_dists:
                # add the pseudocount so probabilities don't zero out
                meme = meme*(.98) + psuedocount
                #norms = np.repeat(np.linalg.norm(meme, axis=1), 4).reshape((-1, 4))
                meme = np.log(meme) - np.log(.25)
                meme_lods.append(meme)
    #make distribution objects
    meme_list = [Meme(distribution, log_odds) for distribution, log_odds in zip(meme_dists, meme_lods)]
    return meme_list

CTCF_memes = process_meme('/home/kal/TF_models/data/memes/CTCF.meme')
mystery_memes = process_meme('/home/kal/TF_models/data/memes/mystery_motif.meme') 
