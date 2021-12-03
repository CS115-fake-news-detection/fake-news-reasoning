import datetime
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('SnowballStemmer')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet



def print_message(*s, condition=True):
    s = ' '.join([str(x) for x in s])
    msg = "[{}] {}".format(datetime.datetime.now().strftime("%b %d, %H:%M:%S"), s)

    if condition:
        print(msg, flush=True)

    return msg


def preprocessed_stop(words):
    stopwords_list = stopwords.words('english')
    clean_words = [word for word in words if word not in stopwords_list ] 
    return (clean_words)


def preprocessed_stem(sentences):
    sno = nltk.stem.SnowballStemmer('english')
    stemmed_words = [sno.stem(word) for word in sentences]
    return stemmed_words

def preprocessed_pos(sentences):
    tags = [] #have the pos tag included
    nava_sen = []
    pt = nltk.pos_tag(sentences)

    nava = []
    nava_words = []
    for t in pt:
#        if t[1].startswith('NN') or t[1].startswith('NNS') or t[1].startswith('NNP') or t[1].startswith('NNPS') or t[1].startswith('JJ') or t[1].startswith('JJR') or t[1].startswith('JJS') or  t[1].startswith('VB') or t[1].startswith('VBG') or t[1].startswith('VBN') or t[1].startswith('VBP') or t[1].startswith('VBZ') or t[1].startswith('RB') or t[1].startswith('RBR') or t[1].startswith('RBS'):

        if t[1].startswith('NN') or t[1].startswith('NNS') or t[1].startswith('NNP') or t[1].startswith('NNPS') or t[1].startswith('JJ') or t[1].startswith('JJR') or t[1].startswith('JJS') or  t[1].startswith('VB') or t[1].startswith('VBG') or t[1].startswith('VBN') or t[1].startswith('VBP') or t[1].startswith('VBZ'):
            nava.append(t)
            nava_words.append(t[0])
    return nava_words
    
    
def find_antonym(word, word_freq):
    synonyms = set()
    antonyms = set()

    for syn in wordnet.synsets(word):   
        for l in syn.lemmas():
            synonyms.add(l.name())
            for antonym in l.antonyms():
                antonyms.add(antonym.name())
                
    antonymsL = list(antonyms)
    if len(antonymsL) != 0:
        selected = list(antonyms)[0]
        freq = 0
        for antonym in antonyms:
            if antonym in word_freq.keys():
                freqnew = int(word_freq[antonym])                
                if freqnew > freq:
                    selected = antonym
                    freq = freqnew
        return(selected)        
    
    else:
        return None

no_neg_list = ['be', 'being', 'just', 'been', 'receive', 'received', 'receiving', 'give', 'gave', 'given', 'even', 'get', 'getting', 'got', 'come', 'came', 'go', 'going', 'went', 'do', 'did', 'done', 'make', 'made', 'call', 'called', 'calling', 'face', 'buy', 'bought', 'mental', 'have', 'had', 'having', 'look', 'looked', 'looking']



def preprocessed_neg(words, word_freq):
    words_org = words.copy()
    idx = 0
    indices = []
    indices = [i for i, x in enumerate(words) if x == 'not' or x == 'never']    
    c = 0
#     print(words)
#     print(indices)
    for i in indices:
        le = len(words)
        i -= c
        
        if i >= 0 and i+1 < le : 
            if words[i+1] not in no_neg_list:
                antonym = find_antonym(words[i+1], word_freq)
                if antonym:            
                    words[i] = antonym
                    words.pop(i+1)
                    c += 1
    #                 print('***********************************')
    # if c > 0:
        
    #     print(' '.join(words_org))
    #     print(' '.join(words))
    #     print('***********************************\n\n')
        
    return (words)


filepath = 'sorted.uk.word.unigrams'  
word_freq = {}  
count = 0
with open(filepath, encoding= 'utf-8') as f:
    for line in f:
        line = line.rstrip()
        if line:
            x = line.split('\t')
            #print(x)
            #print(key, val)
            #print(str(x[1]))
            word_freq[x[1]] = str(x[0])
        count +=1
        if count > 100000:
            break



def clean_str(string):
    # print(f'string:\t {string} ')
    # print(steps)
    
    # add \' for negation process if needed
    # string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    # per Ben's request, numeric will confuse the model; therefore, remove the numeric
    string = re.sub(r"[^A-Za-z\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # retval = string.strip().lower()
    str2 = string.strip().lower()
    words= nltk.word_tokenize(str2)
    
    retval = ' '.join(words)
    return retval
    
def preprocess(string, step = 'none'):
    # print(f'string:\t {string} ')
    # print(steps)
    
    # add \' for negation process if needed
    # string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    # per Ben's request, numeric will confuse the model; therefore, remove the numeric
    string = re.sub(r"[^A-Za-z\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # retval = string.strip().lower()
    str2 = string.strip().lower()
    words= nltk.word_tokenize(str2)
    
      # Need to follow exact sequence of preprocessing steps
    if step == 'all':
        words = preprocessed_neg(words, word_freq)
        words = preprocessed_pos(words)
        words = preprocessed_stop(words)
        words = preprocessed_stem(words)
    else:
        if "neg" in step:
            words = preprocessed_neg(words, word_freq)
        
        if "pos" in step:
            words = preprocessed_pos(words)
        # print("pos word \n", words)
        
        if "stop" in step:
            words = preprocessed_stop(words)
        # print("stop word \n", words)
        
        if "stem" in step:
            words = preprocessed_stem(words)
    # print("stem word \n", words)
    
    retval = ' '.join(words)
    # print(f'Processed:\t{retval}')
    
    return retval