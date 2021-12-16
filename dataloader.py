import re
import nltk
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm
import os

class Dataset():
    def __init__(self, dirname, filename):
        self.dirname = dirname
        self.filename = filename
        self.FILENAME = filename.upper()

    def __iter__(self):
        '''Make Dataset iterable.'''
        # read txts
        for root, dirs, files in os.walk(self.dirname):
            # for each txt
            for file in files:
                if ((self.filename in file or self.FILENAME in file) and
                        file.endswith('.txt') and '._' not in file):
                    print("Processing", file)
                    # generate tokens from txt
                    with open(os.path.join(root, file), 'r') as f:
                        text = f.read()
                        sentences = self.text2sen(text)
                        tokens = self.sen2token(sentences)

                        yield tokens  # iteratable

    def text2sen(self, text):
        '''Split paragraph into sentences.'''
        text_only = re.sub('@@\d+', '',text).replace("<p>", " ").replace("<h>", " ").replace("@", "")
        return nltk.tokenize.sent_tokenize(text_only)

    def sen2token(self, sentences):
        '''given a list of sentence, return a token list after cleaning.
        '''
        clear_tokens = []
        #for sentence in sentences:
        for sentence in tqdm(sentences, total=len(sentences)):
            # split sentence by word
            word_punct_token = WordPunctTokenizer().tokenize(sentence)
            # check each token
            for token in word_punct_token:
                token = self.tokenCleaner(token)  # return "" if token is bad
                if len(token) > 0:
                    clear_tokens.append(token)
        return clear_tokens

    def tokenCleaner(self, token):
        '''clear token, return empty str "" if the token is bad.
        '''
        token = token.lower()
        # remove any value that are not alphabetical
        new_token = re.sub(r'[^a-zA-Z]+', '', token)
        # remove empty value and single character value
        if new_token != "" and len(new_token) >= 2:
            vowels=len([v for v in new_token if v in "aeiou"])
            if vowels != 0: # remove line that only contains consonants
                return new_token
            else:
                return ""
        else:
            return ""
