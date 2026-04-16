import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    drop_words = {"i", "we", "she", "he", "a", "an", "the", "this", "that", "these", "those", "its", "my", "very", "really", "just", "so"} #partially llm generated

    nearby_keys = { # llm generated
        'a': ['s', 'q', 'z'],
        'e': ['w', 'r', 'd'],
        'i': ['u', 'o', 'k'],
        'o': ['i', 'p', 'l'],
        'u': ['y', 'i', 'j'],
        'A': ['S', 'Q', 'Z'],
        'E': ['W', 'R', 'D'],
        'I': ['U', 'O', 'K'],
        'O': ['I', 'P', 'L'],
        'U': ['Y', 'I', 'J'],
    }

    words = word_tokenize(example["text"])
    new_words = []

    for word in words:
        if word.lower() in drop_words:
            if random.random() < 0.35:
                continue
            else:
                new_words.append(word)
        elif random.random() < 0.45:
            synsets = wordnet.synsets(word)
            if not synsets:
                new_words.append(word)
                continue
            if len(synsets[0].lemmas()) > 1:
                new_words.append(synsets[0].lemmas()[1].name().replace("_", " "))
            else:
                new_words.append(word)
        else:    
            new_words.append(word)

    swapped_ex = TreebankWordDetokenizer().detokenize(new_words).replace(" .", ".")

    new_chars = ""
    for char in swapped_ex:
        if char in nearby_keys and random.random() < 0.075:
            new_chars += nearby_keys[char][random.randint(0,2)]
        else:
            new_chars += char

    sentences = new_chars.split(". ")
    random.shuffle(sentences)
    example["text"] = ". ".join(sentences)
    

    ##### YOUR CODE ENDS HERE ######

    return example
