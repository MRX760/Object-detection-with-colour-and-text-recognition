import easyocr
import re
from collections import Counter
import os

def OCR(img, lang='en'):    
    """
    Args:
    img (List): numpy array of img
    lang (str): language for OCR. default = 'id', en=english, etc..
    """
    bboxs, texts = [],[]
    # Initialize the reader with the desired languages
    if type(lang) == str:
        reader = easyocr.Reader([lang])
    else: #if list
        reader = easyocr.Reader(lang)  # You can specify multiple languages
    
    # Perform OCR on an image
    result = reader.readtext(img)
    
    # Print the recognized text
    for (bbox, text, prob) in result:
        bboxs.append(bbox)
        texts.append(text)
    return bboxs,texts

def words(text): return re.findall(r'\w+', text.lower())

try:
    WORDS = Counter(words(open('..\\dictionary\\filtered_sorted_words.txt').read()))
except:
    try:
        WORDS = Counter(words(open(os.path.abspath("dictionary/filtered_sorted_words.txt")).read()))
    except Exception as e:
        raise e
        

def jaccard(word, cands): 
    """
    Probability of candidates based on jaccard similarity in letter level and character shape if there's any word with same jaccard value.
    Args:
    word (string): error word
    cands (list(string)): list of candidate word
    Returns:
    List: list of jaccard value per candidate word
    """
    jaccard = []
    set_word = set(word)
    for i in cands:
        set_cand = set(i)
        
        intersection = len(set_word.intersection(set_cand))
        union = len(set_word.union(set_cand))
        jaccard.append(intersection/union)
    return jaccard
    
def correction(word): 
    "Most probable spelling correction for word."
    candidate_list = list(candidates(word))
    jac = jaccard(word, candidate_list)

    #just in case if there's a value with the same jaccard 
    max_jaccard = max(jac)
    idx = [i for i, val in enumerate(jac) if val == max_jaccard]
    if len(idx)>1:
        for i in idx:
            if len(candidate_list[i])>=len(word):
                return candidate_list[i]
    return candidate_list[idx[0]]
    #to do, try n-gram overlapping method.     
        
def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
    # return (known([word]) or known(edits1(word)) or known(edits2(word)) or known(edits3(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    # transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    # return set(deletes + transposes + replaces + inserts)
    return set(deletes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# def edits3(word):
#     "All edits that are three edits away from `word`."
#     return (e3 for e1 in edits1(word) for e2 in edits1(e1) for e3 in edits1(e2))

def detect_text(img, language='en'):
    word, correction_result = "", ""
    bbox, text = OCR(img, language)
    for i in text:
        word = word + " " + i
    word = word.lower()
    word = word.split()
    for i in word:
        if i.isalpha():
            correction_result = correction_result+ correction(i) + " "
        else:
            correction_result = correction_result+i+" "
    return correction_result