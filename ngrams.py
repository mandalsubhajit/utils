# code used in: https://medium.com/tim-black/fuzzy-string-matching-at-scale-41ae6ac452c2

'''
Sample Usage:

from sklearn.feature_extraction.text import TfidfVectorizer

org_names = names['buyer'].unique()
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
tf_idf_matrix = vectorizer.fit_transform(org_names)

# Then cosine similarity can be applied for getting string similarity.

'''

def ngrams(string, n=3):
    string = fix_text(string) # fix text encoding issues
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    string = string.lower() #make lower case
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string) #remove the list of chars defined above
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single space
    string = ' '+ string +' ' # pad names for ngrams...
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]
