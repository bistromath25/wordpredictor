import re
import unicodedata
import string
import random
import nltk

from nltk.probability import ConditionalFreqDist


class WordPredictor:
    def __init__(self, text_file=None, chain_length=3):
        self.text_file = text_file
        self.text = ''
        self.words = []
        self.chain_length = chain_length
        self.model = {}
    
    def init(self):
        assert self.text_file, "text_file not found"
        self.words = self.filter(self.text)
        self.words = self.clean(self.words)
        self.model = self.generate_ngram_model(self.words, self.chain_length)
    
    def loadtextfile(self, text_file=None):
        self.text_file = text_file
        assert self.text_file, "text_file not found"
        with open(self.text_file, 'r') as text_file:
            while True:
                line = text_file.readline()
                self.text += line
                if not line:
                    break
    
    def setchainlength(self, chain_length=3):
        assert chain_length >= 2, "n must be at least 2"
        self.chain_length = chain_length
    
    def filter(self, text):
        text = (unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore'))
        text = re.sub('<.*?>', ' ', text)
        text = text.translate(str.maketrans(' ', ' ', string.punctuation))
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = re.sub("\n", " ", text)
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    def clean(self, words):
        tokens = nltk.word_tokenize(self.words)
        wnl = nltk.stem.WordNetLemmatizer()
        words = []
        
        for word in tokens:
            words.append(wnl.lemmatize(word))
        
        return words
    
    def generate_ngram_model(self, words, chain_length=3):
        assert chain_length >= 2, 'n must be at least 2'
        self.chain_length = chain_length
        ngrams = list(nltk.ngrams(words, self.chain_length, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        cfdist = ConditionalFreqDist()
        
        #print(ngrams)
        for gram in ngrams:
            #print(gram[:-1], gram[-1])
            cfdist[gram[:-1]][gram[-1]] += 1
        
        for current_words in cfdist:
            total_count = float(sum(cfdist[current_words].values()))
            for next_word in cfdist[current_words]:
                cfdist[current_words][next_word] /= total_count
        
        #print(cfdist)
        return cfdist
    
    def predict(self, prompt):
        prompt = self.filter(prompt).split()
        prev_words = prompt[len(prompt) - self.chain_length + 1:len(prompt)]
        
        prediction = sorted(dict(self.model[prev_words[0], prev_words[1]]), key=lambda x: dict(self.model[prev_words[0], prev_words[1]])[x], reverse=True)
        
        #print(f'prediction: {prediction}')
        
        words = []
        weight = []
        for key, prob in dict(self.model[prev_words[0], prev_words[1]]).items():
            words.append(key)
            weight.append(prob)
        
        try:
            next_word = random.choices(words, weights=weight, k=1)[0]
        except:
            next_word = random.choice(self.words)
        
        return next_word
