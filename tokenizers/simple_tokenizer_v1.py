import re 

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab 
        self.int_to_str = {i:s for i,s in enumerate(vocab)}
    
    def encode(self, text):
        preprocessed = re.split(r'[,;:.()?!"\']|--|\s', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        token_ids = [self.str_to_int[s] for s in preprocessed]        
        return token_ids
    
    def decode(self, token_ids):
        text = " ".join([self.int_to_str[id] for id in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) 
        return text
        