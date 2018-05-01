import numpy as np
try:
    import jellyfish

    def hamming_dist(s1, s2):
        if s1 == s2:
            return(0)
        else:
            return(float(jellyfish.hamming_distance(s1, s2)))
except:
    def hamming_dist(seq1, seq2):
        if s1 == s2:
            return(0)
        else:
            return(float(sum(x != y for x, y in zip(seq1, seq2))))

# A2M reference:
# https://compbio.soe.ucsc.edu/a2m-desc.html
# Amino acid alphabet:
AAamb_ORDER = 'ABCDEFGHIKLMNPQRSTVWXYZ-'
AAamb_LIST = list(AAamb_ORDER)
AAamb_DICT = {c:i for i, c in enumerate(AAamb_LIST)}
AAamb_SET = set(AAamb_LIST)
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY-'
AA_LIST = list(AA_ORDER)
AA_DICT = {c:i for i, c in enumerate(AA_LIST)}
AA_SET = set(AA_LIST)


def prune_a2m(seq):
    '''Drop columns that are not part of the alignment (A2M format).'''
    return(''.join([s for s in seq if s in AAamb_SET]))


def filter_seq(seq):
    '''Filter away ambiguous character containing sequences.'''
    if set(list(seq)) <= AA_SET:
        return(seq)
    else:
        return(None)


def index_a2m(seqs):
    '''Index a A2M alignment.'''
    li = len(prune_a2m(seqs[0]))
    for seq in seqs:
        assert(li == len(prune_a2m(seq)))
    return(list(range(li)))


def seq2onehot(seq):
    '''Translate a sequence string into one hot encoding.'''
    out = np.zeros((len(AA_DICT), len(seq)))
    for i, s in enumerate(seq):
        out[AA_DICT[s]][i] = 1
    return(out)


class VAEdata():
    '''Object to contain all data for VAE training/testing.'''
    def __init__(self, name):
        self.name = name

        # Train data:
        self.__read_train_set_filename = None
        self.__read_train_set_fformat = None
        self.__read_train_set_num_reads = None
        self.train_seq_dict = None
        self.train_seq_dict_order = None
        self.train_seq_dict_weights = None
        self.train_Neff = None
        self.train_onehot_list = None
        self.train_onehot_order = None

        # Test data:
        self.__read_test_set_filename = None
        self.__read_test_set_fformat = None
        self.__read_test_set_offsett = None
        self.__wt_seq_inp = None  # This copy is necessary for reconstruction
        self.wt_seq = None
        self.test_seq_dict = None
        self.test_value_list = None
        self.test_seq_dict_order = None
        self.test_onehot_list = None
        self.onehot_order = None

        # Column observation cutoff:
        self.obs_cutoff_min_freq = 0.05
        self.obs_cutoff_sele = None

    def set_wt_set(self, wt_seq):
        '''Set the wild type sequence, which works as a reference point for the mutation data.'''
        if wt_seq != self.__wt_seq_inp:
            self.wt_seq = wt_seq
            self.__wt_seq_inp = wt_seq
            # Update mut_code test set:
            if self.__read_test_set_fformat is not None and self.__read_test_set_fformat == 'mut_code':
                self.read_test_set(self.__read_test_set_filename)
        else:
            pass

    def get_seq_len(self):
        '''Return the length of sequences in the alignment.'''
        ltrain = None
        ltest = None
        if self.train_seq_dict is not None and self.test_seq_dict is not None:
            ltrain = len(self.train_seq_dict[self.train_seq_dict_order[0]])
            ltest = len(self.test_seq_dict[self.test_seq_dict_order[0]])
            assert(ltrain == ltest)
            return(ltrain)
        elif self.train_seq_dict is not None:
                ltrain = len(self.train_seq_dict[self.train_seq_dict_order[0]])
                return(ltrain)
        elif self.test_seq_dict is not None:
            ltest = len(self.test_seq_dict[self.test_seq_dict_order[0]])
            return(ltest)
        else:
            return(None)

    def print_train_set_seq(N=10):
        '''Print sequences from the training set.'''
        if self.train_seq_dict is not None:
            for i in range(N):
                print('{}'.format(self.train_seq_dict[self.train_seq_dict_order[i]][-1]))
        else:
            return(None)

    def read_train_set(self, filename, fformat=None, num_reads=None):
        '''Read an alignment file of either fasta or A2M format.'''
        from Bio import SeqIO
        self.__read_train_set_filename = filename
        # Load previous settings if nothing provided:
        if fformat is None:
            fformat = self.__read_train_set_fformat
        else:
            self.__read_train_set_fformat = fformat
        if fformat is None:
            raise RuntimeError('"fformat" must be defined.')
        if num_reads is None:
            num_reads = self.__read_train_set_num_reads
        else:
            self.__read_train_set_num_reads = num_reads
        seq_dict = dict()
        seq_dict_order = dict()        
        i = 0
        seqlen = False  # All sequences must be equally long
        for record in SeqIO.parse(filename, 'fasta'):
            seq = str(record.seq)
            ID = str(record.id)
            if num_reads is not None and i == num_reads:
                break
            if fformat.lower() == 'a2m':
                seq = prune_a2m(seq)
            if seqlen is False:
                seqlen = len(seq)
            elif seqlen != len(seq):
                raise RuntimeError('Sequences are not equally long in alignment. number: {}, sequence: {}, ID: {}'.format(i, seq, ID))
            # Current, filtering is enforced; other functions depend on the list of amino acids being immutable:
            seq = filter_seq(seq)
            if seq is not None:
                assert(ID not in seq_dict)
                assert(set(list(seq)) <= AA_SET)
                seq_dict[ID] = seq
                seq_dict_order[i] = ID
                i += 1
        # Apply cutoff on per-column observations:
        self.train_seq_dict = self.__obs_cutoff(seq_dict, train_set=True)
        self.train_seq_dict_order = seq_dict_order
        # Whenever the training data change it may affect the test data as well:
        if self.test_seq_dict is not None:
            self.read_test_set(self.__read_test_set_filename)
        # Default sequence weight is 1:
        if self.train_seq_dict_weights is None or len(self.train_seq_dict_weights) != len(seq_dict):
            self.train_seq_dict_weights = {k:1 for k in seq_dict.keys()}
            self.train_Neff = float(len(seq_dict.keys()))
        # Make onehot repressentation:
        self.train_onehot_list, self.train_onehot_order = self.__make_onehot(self.train_seq_dict)

    def reweight_sequences(self, theta=0.2, verbose=True, cached_file=None):
        '''Compute new weights for the sequences based on similarity threshold theta.'''
        if cached_file is not None:
            import pickle
            try:
                with open(cached_file, 'rb') as fh:
                    self.train_seq_dict_weights = pickle.load(fh)
                weights = np.array(list(self.train_seq_dict_weights.values()))
                assert(len(weights) == len(self.train_seq_dict))
                print('Loaded weights from filename: {}'.format(cached_file))
                self.train_Neff = sum(1 / weights)
                return
            except:
                print('Could not load weights, either no file or wrong format.')
                pass
        import time
        Nseq = len(self.train_seq_dict)
        seq_len = self.get_seq_len()
        weights = np.array([1.0] * Nseq)
        start = time.process_time()
        for i in range(Nseq):
            if i%1000 == 0 and verbose:  # Print progress
                print('{}/{} sequences processed.\nTime: {}'.format(i, Nseq, str(time.process_time()-start)))
            for j in range(i+1, Nseq):
                if (hamming_dist(self.train_seq_dict[self.train_seq_dict_order[i]], self.train_seq_dict[self.train_seq_dict_order[j]]) / seq_len) < theta:
                    weights[i] += 1
                    weights[j] += 1
        self.train_seq_dict_weights = {self.train_seq_dict_order[i]: 1/weights[i] for i in range(Nseq)}
        self.train_Neff = sum(1 / weights)
        if cached_file is not None:
            print('Dumped weights to filename: {}'.format(cached_file))
            with open(cached_file, 'wb') as fh:
                pickle.dump(self.train_seq_dict_weights, fh)

    def read_test_set(self, filename, fformat=None, wt_seq=None, offset=None):
        '''
        Makes test set by reading semicolon separated file, last column contains the measurements,
        first colum contains either a mutation codes (mut_code) e.g. E121H or a fasta seqeunces.
        '''
        self.__read_test_set_filename = filename
        # Load previous settings if nothing provided:
        if fformat is None:
            fformat = self.__read_test_set_fformat
        else:
            self.__read_test_set_fformat = fformat
        if wt_seq is None:
            # Use restored wt_seq:
            wt_seq = self.__wt_seq_inp
        else:
            # Provided as input, then update wt_seq:
            self.__wt_seq_inp = wt_seq
            self.wt_seq = wt_seq
        if offset is None:
            offset = self.__read_test_set_offset
        else:
            self.__read_test_set_offsett = offset
        # Read the file according to format:
        if fformat == 'mut_code':
            if wt_seq is None or offset is None:
                raise RuntimeError('File format specified as "mut_code", wt_seq and offset must be specified.')
            self.__read_mut_code_test_set()
        elif fformat == 'fasta':
            self.__read_fasta_test_set()
        else:
            raise RuntimeError('Cannot recognize "fformat" option:', fformat)

        # Apply cutoff on per-column observations:
        if self.train_seq_dict is not None:
            self.test_seq_dict = self.__obs_cutoff(self.test_seq_dict, train_set=False)
        # Make onehot repressentation:
        self.test_onehot_list, self.test_onehot_order = self.__make_onehot(self.test_seq_dict)

    def __read_fasta_test_set(self):
        seq_dict = dict()
        value_list = list()
        seq_dict_order = dict()
        i = 0
        seqlen = False  # All sequences must be equally long
        with open(self.__read_test_set_filename, 'r') as fh:
            for l in fh:
                l = l.strip()
                if l.startswith('#'):
                    continue
                cols = l.split(';')
                seq = cols[0]
                if seqlen is False:
                    seqlen = len(seq)
                elif seqlen != len(seq):
                    raise RuntimeError('Sequences are not equally long in alignment. number: {}, sequence: {}, ID: {}'.format(i, seq, ID))
                value_list.append(float(cols[-1]))
                key = i
                seq_dict[key] = value
                seq_dict_order[i] = i  # For consistency with the "mut_code" format
                i += 1
        self.test_seq_dict = seq_dict
        self.test_value_list = value_list
        self.test_seq_dict_order = seq_dict_order

    def __read_mut_code_test_set(self):
        '''
        Read semicolon separated mutation data.
        Format looks like this:
        # Reference: Melamed et al., RNA 2013 (Supplementary Table 5), PMID: 24064791
        # Experimental data columns: XY_Enrichment_score
        mutant;effect_prediction_epistatic;effect_prediction_independent;XY_Enrichment_score
        G169W,F170V;-18.1610034815;-15.1524950321;0.059160001
        .
        .
        '''
        import re
        seq_dict = dict()
        value_list = list()
        seq_dict_order = dict()
        first_line = True
        i = 0
        with open(self.__read_test_set_filename, 'r') as fh:
            for l in fh:
                l = l.strip()
                if l.startswith('#'):
                    continue
                # Skip the first line:
                elif first_line:
                    first_line = False
                    cols = l.split(';')
                    print('Header was encountered:\n{}\nMutation key is grabbed from this column: "{}"\nEffect is grabbed from this column: "{}"'.format(l, cols[0], cols[-1]))
                    continue
                cols = l.split(';')
                value_list.append(float(cols[-1]))
                mutkey = tuple(tuple(re.split(r'(\d+)', sp)) for sp in cols[0].split(','))
                wt_cp = list(self.wt_seq)  # Make the wild type sequence mutable
                for k in mutkey:
                    assert(k[0] == wt_cp[int(k[1]) - self.__read_test_set_offsett])
                    # Introduce the mutation:
                    wt_cp[int(k[1]) - self.__read_test_set_offsett] = k[2]
                seq_dict[mutkey] = ''.join(wt_cp)
                seq_dict_order[i] = mutkey
                i += 1
        self.test_seq_dict = seq_dict
        self.test_value_list = value_list
        self.test_seq_dict_order = seq_dict_order

    def __make_onehot(self, seq_dict):
        '''Translate a sequence strings into one hot encodings.'''
        onehot_order = dict()
        onehot_list = np.zeros((len(seq_dict), len(AA_DICT), self.get_seq_len()))
        for i, item in enumerate(seq_dict.items()):
            ID, seq = item
            onehot_order[ID] = i
            for j, s in enumerate(seq):
                onehot_list[i][AA_DICT[s]][j] = 1
        return(np.array([np.array(list(sample.flatten())).T for sample in onehot_list]), onehot_order)

    def obs_cutoff(self, min_freq):
        '''Set cutoff for dropping columns with too few observations.'''
        self.obs_cutoff_min_freq = min_freq
        # Reload sequences and apply the cutoff:
        if self.train_seq_dict is not None:
            self.read_train_set(self.__read_train_set_filename)
        if self.train_seq_dict is not None and self.test_seq_dict is not None:
            self.read_test_set(self.__read_test_set_filename)

    def __obs_cutoff(self, seq_dict, train_set=True):
        '''Drop columns with fewer observations than cutoff.'''
        # Update columns selected if train set:
        if train_set:
            max_gap_freq = 1 - self.obs_cutoff_min_freq
            sequences = list(seq_dict.values())
            # Find the gap frequencies:
            gap_freq = np.array([0.0 for s in sequences[0]])
            for seq in sequences:
                gaps = np.array([1 if s == '-' else 0 for s in seq])
                try:
                    gap_freq += gaps
                except ValueError as e:
                    print('Looks like the alignment format is corrupted. Sequences not equally long?', e)
            gap_freq /= len(sequences)
            # Select all columns above the threshold:
            self.obs_cutoff_sele = gap_freq <= max_gap_freq
        elif self.__wt_seq_inp is not None:
            self.wt_seq = ''.join([nt for keep, nt in zip(self.obs_cutoff_sele, self.__wt_seq_inp) if keep])

        # Update the sequences:
        for k, seq in seq_dict.items():
            seq_dict[k] = ''.join([nt for keep, nt in zip(self.obs_cutoff_sele, seq) if keep])
        return(seq_dict)


#Compute log probability of a particular mutant sequence from a pwm and a one-hot encoding
def compute_log_probability(one_hot_seq, pwm):
    prod_mat = np.matmul(one_hot_seq.T, pwm)
    log_prod_mat = np.log(prod_mat)
    sum_diag = np.trace(log_prod_mat)
    return(sum_diag)


def PWM_MAP_seq(pwm):
    '''Compute the most likely protein sequence (MAP estimate) given a position weight matrix (PWM)'''
    MAP = np.argmax(pwm, axis=0)
    return(''.join([ORDER_LIST[m] for m in MAP]))

    