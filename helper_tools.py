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
AAamb_LIST = list(AAmb_ORDER)
AAamb_DICT = {c:i for i, c in enumerate(AAamb_ORDER)}
AAamb_SET = set(AAamb_ORDER)
AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY-'
AA_LIST = list(AA_ORDER)
AA_DICT = {c:i for i, c in enumerate(AA_ORDER)}
AA_SET = set(AA_ORDER)


def prune_a2m(seq):
    '''Drop columns that are not part of the alignment (A2M format).'''
    return ''.join([s for s in seq if s in AAamb_SET])


def filter_seq(seq):
    '''Filter away ambiguous character containing sequences'''
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


def seq2onehot(seq, aa_dict):
    '''Translate a sequence string into one hot encoding'''
    out = np.zeros((len(aa_dict), len(seq)))
    for i, s in enumerate(seq):
        out[aa_dict[s]][i] = 1
    return out


def read_mutation_data(filename):
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
    with open(filename, 'r') as fh:
        data_dict = dict()
        for l in fh:
            if l.startswith('#'):
                continue
            elif len(data_dict) == 0:
                data_dict = {el:[] for el in l.split(';')}
                h2p = {i:el for i, el in enumerate(l.split(';'))}
                continue
            for i, el in enumerate(l.split(';')):
                if i == 0:
                    el = e
                data_dict[h2p[i]].append(el)
    return data_dict


def read_MSA(filename, fformat='fasta', num_reads=None):
    '''Read an alignment file of either fasta or A2M format.'''
    from Bio import SeqIO
    # ids = list()
    # seqs = list()
    seq_dict = dict()
    i = 0
    for record in SeqIO.parse(filename, 'fasta'):
        if num_reads not None and i == num_reads:
            break
        else:
            i += 1

        if fformat == 'a2m':
            record.seq = prune_a2m(record.seq)
            # Current, filtering is enforced because other functions also depend on the list of amino acids being immutable:
            record.seq = filter_seq(record.seq)
        if record.seq is not None:
            # ids.append(record.id)
            # seqs.append(records.seq)
            assert(record.id not in seq_dict)
            seq_dict[record.id] = records.seq
    # return ids, seqs
    return seq_dict


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
    
    