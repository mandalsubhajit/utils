# calculation of idf from scratch without requiring any external libraries
# 35% faster than standard libraries when only idf is required

'''
Sample Usage: 
textList = ['Lorem ipsum dolor sit amet', 'consectetur adipiscing elit', 'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua']*15000
get_idf(textList)
'''

import re
import math

def get_idf(textList):
    token_pattern = re.compile(r'(?u)\b\w\w+\b')
    N = len(textList)
    idf_dict = {}
    
    def _update_vocab(text):
        tokens = token_pattern.findall(text)
        tokens = [t.lower() for t in tokens]
        for t in tokens:
            if t in idf_dict:
                idf_dict[t] += 1
            else:
                idf_dict[t] = 1
    
    _ = [_update_vocab(text) for text in textList]
    idf_dict = {token: 1 + math.log((1+N)/(1+idf_dict[token])) for token in idf_dict}
        
    return idf_dict
