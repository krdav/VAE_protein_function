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
        self.read_MSA_filename = None
        self.read_MSA_fformat = None
        self.read_MSA_num_reads = None
        self.seq_dict = None
        self.seq_dict_order = None
        self.obs_cutoff_applied = False
        self.obs_cutoff_min_freq = None
        self.obs_cutoff_sele = None
        self.seq_dict_weights = None
        self.Neff = None
        self.onehot_list = None
        self.onehot_order = None
        self.wt_seq = None
        self.wt_seq_inp = None
        self.read_mutation_data_filename = None
        self.make_mutants_offsett = None
        self.mut_data_dict = None
        self.mut_seq_dict = None

    def read_MSA(self, filename, fformat='fasta', num_reads=None):
        '''Read an alignment file of either fasta or A2M format.'''
        self.read_MSA_filename = filename
        self.read_MSA_fformat = fformat
        self.read_MSA_num_reads = num_reads
        from Bio import SeqIO
        seq_dict = dict()
        seq_dict_order = dict()
        seq_dict_weights = dict()
        i = 0
        seqlen = False
        for record in SeqIO.parse(filename, 'fasta'):
            seq = str(record.seq)
            ID = str(record.id)
            if seqlen is False:
                seqlen = len(seq)
            elif seqlen != len(seq):
                raise RuntimeError('Sequences are not equally long in alignment. number: {}, sequence: {}, ID: {}'.format(i, seq, ID))
            if num_reads is not None and i == num_reads:
                break
            if fformat.lower() == 'a2m':
                seq = prune_a2m(seq)
                # Current, filtering is enforced because other functions also depend on the list of amino acids being immutable:
                seq = filter_seq(seq)
            if seq is not None:
                assert(ID not in seq_dict)
                assert(set(list(seq)) <= AA_SET)
                seq_dict[ID] = seq
                seq_dict_order[i] = ID
                seq_dict_weights[ID] = 1  # Default sequence weight is 1
                i += 1
        self.seq_dict = seq_dict
        self.seq_dict_order = seq_dict_order
        self.seq_dict_weights = seq_dict_weights

    def obs_cutoff(self, seq_dict=None, min_freq=0.05, sele=None):
        '''Drop columns with fewer observations than cutoff.'''
        if seq_dict is None:
            assert(self.seq_dict is not None)
        if sele is None:
            if self.obs_cutoff_applied:
                self.read_MSA(self.read_MSA_filename, fformat=self.read_MSA_fformat, num_reads=self.read_MSA_num_reads)
            seq_dict = self.seq_dict
            max_gap_freq = 1 - min_freq
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
            sele = gap_freq <= max_gap_freq
            # Update the sequences:
            for k, seq in seq_dict.items():
                seq_dict[k] = ''.join([nt for keep, nt in zip(sele, seq) if keep])
            self.seq_dict = seq_dict
            self.obs_cutoff_applied = True
            self.obs_cutoff_min_freq = min_freq
            self.obs_cutoff_sele = sele
            # Update all other sequences specified:
            if self.onehot_list is not None:
                self.make_onehot()
            if self.wt_seq_inp is not None:
                self.wt_seq = ''.join([nt for keep, nt in zip(sele, self.wt_seq_inp) if keep])
            if self.mut_seq_dict is not None:
                self.make_mutants(wt_seq=self.wt_seq_inp, offset=self.make_mutants_offsett, filename=self.read_mutation_data_filename)
        else:
            assert(sele is not None)
            for k, seq in seq_dict.items():
                seq_dict[k] = ''.join([nt for keep, nt in zip(sele, seq) if keep])
            return(seq_dict)
        
    def reweight_sequences(self, theta=0.2, verbose=True, cached_file=None):
        '''Compute new weights for the sequences based on similarity threshold theta.'''
        if cached_file is not None:
            import pickle
            try:
                with open(cached_file, 'rb') as fh:
                    self.seq_dict_weights = pickle.load(fh)
                weights = np.array(list(self.seq_dict_weights.values()))
                assert(len(weights) == len(self.seq_dict))
                print('Loaded weights from filename: {}'.format(cached_file))
                self.Neff = sum(1 / weights)
                return
            except:
                print('Could not load weights, either no file or wrong format.')
                pass
        import time
        Nseq = len(self.seq_dict)
        seq_len = len(self.seq_dict[self.seq_dict_order[0]])
        weights = np.array([1.0] * Nseq)
        start = time.process_time()
        for i in range(Nseq):
            if i%1000 == 0 and verbose:  # Print progress
                print('{}/{} sequences processed.\nTime: {}'.format(i, Nseq, str(time.process_time()-start)))
            for j in range(i+1, Nseq):
                if (hamming_dist(self.seq_dict[self.seq_dict_order[i]], self.seq_dict[self.seq_dict_order[j]]) / seq_len) < theta:
                    weights[i] += 1
                    weights[j] += 1
        self.seq_dict_weights = {self.seq_dict_order[i]: 1/weights[i] for i in range(Nseq)}
        self.Neff = sum(1 / weights)
        if cached_file is not None:
            print('Dumped weights to filename: {}'.format(cached_file))
            with open(cached_file, 'wb') as fh:
                pickle.dump(self.seq_dict_weights, fh)

    def make_onehot(self, filename=None, fformat='fasta', num_reads=None):
        '''Translate a sequence strings into one hot encodings.'''
        if self.seq_dict is None and filename is None:
            raise RuntimeError('Need either a filename or filled a "mut_data_dict".')
        elif self.seq_dict is None:
            self.read_MSA(filename, fformat=fformat, num_reads=None)
        onehot_order = dict()
        onehot_list = np.zeros((len(self.seq_dict), len(AA_DICT), len(list(self.seq_dict.values())[0])))
        for i, item in enumerate(self.seq_dict.items()):
            ID, seq = item
            onehot_order[ID] = i
            for j, s in enumerate(seq):
                onehot_list[i][AA_DICT[s]][j] = 1
        self.onehot_order = onehot_order
        self.onehot_list = onehot_list

    def read_mutation_data(self, filename):
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
        self.read_mutation_data_filename = filename
        with open(filename, 'r') as fh:
            mut_data_dict = None
            for l in fh:
                l = l.strip()
                if l.startswith('#'):
                    continue
                # Skip the first line:
                elif mut_data_dict is None:
                    mut_data_dict = dict()
                    cols = l.split(';')
                    print('Header was encountered:\n{}\nMutation key is grabbed from this column: "{}"\nEffect is grabbed from this column: "{}"'.format(l, cols[0], cols[-1]))
                    continue
                cols = l.split(';')
                key = tuple(tuple(re.split(r'(\d+)', sp)) for sp in cols[0].split(','))
                value = float(cols[-1])
                mut_data_dict[key] = value
        self.mut_data_dict = mut_data_dict

    def set_wt_set(self, wt_seq):
        self.wt_seq = wt_seq
        self.wt_seq_inp = wt_seq
        
    def make_mutants(self, wt_seq=None, offset=0, filename=None):
        '''Introduce mutations into provided wild type sequence.'''
        self.make_mutants_offsett = offset
        if self.mut_data_dict is None and filename is None:
            raise RuntimeError('Need either a filename or filled a "mut_data_dict".')
        elif filename is not None:
            self.read_mutation_data(filename)
        # The input wt_seq updates the one in object:
        if wt_seq is None and self.wt_seq is None:
            raise RuntimeError('Need a wt_seq.')
        elif wt_seq is None:
            wt_seq = self.wt_seq
        elif self.wt_seq is None:
            self.wt_seq = wt_seq
            self.wt_seq_inp = wt_seq
        mut_seq_dict = dict()
        for mutkey in self.mut_data_dict.keys():
            wt_cp = list(wt_seq)
            for k in mutkey:
                assert(k[0] == wt_cp[int(k[1]) - offset])
                wt_cp[int(k[1]) - offset] = k[2]
            mut_seq_dict[mutkey] = ''.join(wt_cp)
        if self.obs_cutoff_applied:
            self.mut_seq_dict = self.obs_cutoff(seq_dict=mut_seq_dict, sele=self.obs_cutoff_sele)
        else:
            self.mut_seq_dict = mut_seq_dict


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

    