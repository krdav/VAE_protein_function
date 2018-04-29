import numpy as np

try:
    import jellyfish

    def hamming_dist(s1, s2):
        if s1 == s2:
            return 0
        else:
            return jellyfish.hamming_distance(unicode(s1), unicode(s2))
except:
    def hamming_dist(seq1, seq2):
        return sum(x != y for x, y in zip(seq1, seq2))


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
    return ''.join([s for s in seq if s in AAamb_SET])


def filter_seq(seq):
    '''Filter away ambiguous character containing sequences.'''
    if set(list(seq)) <= AA_SET:
        return seq
    else:
        return None


def index_a2m(seqs):
    '''Index a A2M alignment.'''
    l = len(prune_a2m(seqs[0]))
    for seq in seqs:
        assert(l == len(prune_a2m(seq)))
    return list(range(l))


def seq2onehot(seq):
    '''Translate a sequence string into one hot encoding.'''
    out = np.zeros((len(AA_DICT), len(seq)))
    for i, s in enumerate(seq):
        out[AA_DICT[s]][i] = 1
    return out


class VAEdata():
    def __init__(self, name):
        self.name = name
        self.wt_seq = None
        self.mut_data_dict = None
        self.mut_seq_dict = None
        self.seq_dict = None
        self.seq_dict_order = None
        self.onehot_list = None
        self.onehot_order = None

    def read_mutation_data(self, filename):
        import re
        '''
        Read semicolon separated mutation data.
        Format looks like this:
        # Reference: Melamed et al., RNA 2013 (Supplementary Table 5), PMID: 24064791
        # Experimental data columns: XY_Enrichment_score
        mutant;effect_prediction_epistatic;effect_prediction_independent;XY_Enrichment_score
        G169W,F170V;-18.1610034815;-15.1524950321;0.059160001
        .
        .
        
        with open(filename, 'r') as fh:
            mut_data_dict = dict()
            for l in fh:
                l = l.strip()
                if l.startswith('#'):
                    continue
                elif len(mut_data_dict) == 0:
                    mut_data_dict = {el:[] for el in l.split(';')}
                    h2p = {i:el for i, el in enumerate(l.split(';'))}
                    continue
                for i, el in enumerate(l.split(';')):
                    if i == 0:
                        el = {tuple(re.split(r'(\d+)', sp)) for sp in el.split(',')}
                    mut_data_dict[h2p[i]].append(el)
        self.mut_data_dict = mut_data_dict
        '''

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
                    print('Header was encountered:\n{}\nMutation key is grabbed from this column:{}\nEffect is grabbed from this column:{}'.format(l, cols[0], cols[-1]))
                    continue
                cols = l.split(';')
                key = tuple(tuple(re.split(r'(\d+)', sp)) for sp in cols[0].split(','))
                value = float(cols[-1])
                mut_data_dict[key] = value
        self.mut_data_dict = mut_data_dict

    def make_mutants(self, wt_seq=None, offset=0, filename=None):
        '''Introduce mutations into provided wild type sequence.'''
        if self.mut_data_dict is None and filename is None:
            raise RuntimeError('Need either a filename or filled a "mut_data_dict".')
        elif self.mut_data_dict is None:
            self.read_mutation_data(filename)
        # The input wt_seq updates the one in object:
        if wt_seq is None and self.wt_seq is None:
            raise RuntimeError('Need a wt_seq.')
        elif wt_seq is None:
            wt_seq = self.wt_seq
        elif self.wt_seq is None:
            self.wt_seq = wt_seq
        mut_seq_dict = dict()
        for mutkey in self.mut_data_dict.keys():
            wt_cp = list(wt_seq)
            for k in mutkey:
                assert(k[0] == wt_cp[int(k[1]) - offset])
                wt_cp[int(k[1]) - offset] = k[2]
            mut_seq_dict[mutkey] = ''.join(wt_cp)
        self.mut_seq_dict = mut_seq_dict

    def read_MSA(self, filename, fformat='fasta', num_reads=None):
        '''Read an alignment file of either fasta or A2M format.'''
        from Bio import SeqIO
        seq_dict = dict()
        seq_dict_order = dict()
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
                i += 1
        self.seq_dict = seq_dict
        self.seq_dict_order = seq_dict_order

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


#Compute log probability of a particular mutant sequence from a pwm and a one-hot encoding
def compute_log_probability(one_hot_seq, pwm):
    prod_mat = np.matmul(one_hot_seq.T, pwm)
    log_prod_mat = np.log(prod_mat)
    sum_diag = np.trace(log_prod_mat)
    return sum_diag


def PWM_MAP_seq(pwm):
    '''Compute the most likely protein sequence (MAP estimate) given a position weight matrix (PWM)'''
    MAP = np.argmax(pwm, axis=0)
    return ''.join([ORDER_LIST[m] for m in MAP])


#Compute a new weight for sequence based on similarity threshold theta 
def reweight_sequences(dataset, theta):
    weights=[1.0 for i in range(len(dataset))]
    start = time.process_time()

    for i in range(len(dataset)):

        if i%250==0:
            print(str(i)+" took "+str(time.process_time()-start) +" s ")
            start = time.process_time()

        for j in range(i+1,len(dataset)):
            if hamming_dist(dataset[i], dataset[j])*1./len(dataset[i]) <theta:
                weights[i]+=1
                weights[j]+=1
    return list(map(lambda x:1./x, weights))
    
    