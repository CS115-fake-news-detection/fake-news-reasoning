import datetime
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('SnowballStemmer')
nltk.download('averaged_perceptron_tagger')



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
        if t[1].startswith('NN') or t[1].startswith('NNS') or t[1].startswith('NNP') or t[1].startswith('NNPS') or t[1].startswith('JJ') or t[1].startswith('JJR') or t[1].startswith('JJS') or  t[1].startswith('VB') or t[1].startswith('VBG') or t[1].startswith('VBN') or t[1].startswith('VBP') or t[1].startswith('VBZ') or t[1].startswith('RB') or t[1].startswith('RBR') or t[1].startswith('RBS'):
            nava.append(t)
            nava_words.append(t[0])
    return nava_words

# steps = ["pos","stop","stem"]
steps = []

def clean_str(string):
    # print(f' string \n {string} ')
    # print(steps)
    
    # add \' for negation process if needed
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # retval = string.strip().lower()
    str2 = string.strip().lower()
    words= nltk.word_tokenize(str2)
    
      # Need to follow exact sequence of preprocessing steps
    
    # if "neg" in steps:
    # main_data, snippets_data = preprocessed_neg(main_data, snippets_data)
    
    if "pos" in steps:
        words = preprocessed_pos(words)
    # print("pos word \n", words)
    
    if "stop" in steps:
        words = preprocessed_stop(words)
    # print("stop word \n", words)
    
    if "stem" in steps:
        words = preprocessed_stem(words)
    # print("stem word \n", words)
    
    retval = ' '.join(words)
    # print(retval)
    
    return retval