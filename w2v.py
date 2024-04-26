
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec
import re

# Preprocessing WordNet data
def get_sentences_from_synsets(synsets):
    sentences = []
    for synset in synsets:
        words = [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
        sentence = ' '.join(words)
        sentences.append(sentence)
    return sentences

def preprocess_data(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        # Remove special characters and convert to lowercase
        cleaned_sentence = re.sub(r'[^a-zA-Z\s]', '', sentence).lower()
        cleaned_sentences.append(cleaned_sentence.split())
    return cleaned_sentences

# Retrieve synsets and preprocess data
synsets = wn.all_synsets()
sentences = get_sentences_from_synsets(synsets)
cleaned_sentences = preprocess_data(sentences)

# Training Word2Vec model
vector_size = 100  # Dimensionality of word vectors
window_size = 5  # Context window size
min_word_count = 1  # Minimum word count threshold
sg = 0  # Training algorithm: 0 for CBOW, 1 for skip-gram

word2vec_model = Word2Vec(cleaned_sentences, vector_size=vector_size, window=window_size,
                          min_count=min_word_count, sg=sg)

# Save the Word2Vec model
word2vec_model.save('word2vec_wordnet_model.bin')
