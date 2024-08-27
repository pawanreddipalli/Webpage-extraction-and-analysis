<<<<<<< HEAD
import requests
from gensim import corpora, models
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

url = 'http://www.columbia.edu/~fdc/sample.html'
response = requests.get(url)
text = response.text

sentences = text.split('.')

stop_words = set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of', 'and'])
sentences_preprocessed = []
for sentence in sentences:
    words = sentence.strip().lower().split()
    words_filtered = [word for word in words if word not in stop_words]
    sentences_preprocessed.append(words_filtered)

dictionary = corpora.Dictionary(sentences_preprocessed)

bow_corpus = [dictionary.doc2bow(sentence) for sentence in sentences_preprocessed]

# Train the LSA model using the bag-of-words representation of the preprocessed sentences
lsa_model = models.LsiModel(bow_corpus, num_topics=3, id2word=dictionary)

# Get the vector representation of the preprocessed sentences based on the LSA model
lsa_vectors = []
for sentence in sentences_preprocessed:
    lsa_vector = np.array([tup[1] for tup in lsa_model[dictionary.doc2bow(sentence)]])
    lsa_vectors.append(lsa_vector)

# Print the vector representation of the preprocessed sentences
print('LSA Vectors of the model:')
for i, vector in enumerate(lsa_vectors):
    print(f'Sentence {i+1}: {vector.round(3)}')
    
# Get the top 3 topics based on the LSA model
lsa_topics = lsa_model.print_topics(num_topics=3)
print('\nTop 3 Topics of the model :')
for topic in lsa_topics:
    print(topic)
=======
import requests
from gensim import corpora, models
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

url = 'http://www.columbia.edu/~fdc/sample.html'
response = requests.get(url)
text = response.text

sentences = text.split('.')

stop_words = set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of', 'and'])
sentences_preprocessed = []
for sentence in sentences:
    words = sentence.strip().lower().split()
    words_filtered = [word for word in words if word not in stop_words]
    sentences_preprocessed.append(words_filtered)

dictionary = corpora.Dictionary(sentences_preprocessed)

bow_corpus = [dictionary.doc2bow(sentence) for sentence in sentences_preprocessed]

# Train the LSA model using the bag-of-words representation of the preprocessed sentences
lsa_model = models.LsiModel(bow_corpus, num_topics=3, id2word=dictionary)

# Get the vector representation of the preprocessed sentences based on the LSA model
lsa_vectors = []
for sentence in sentences_preprocessed:
    lsa_vector = np.array([tup[1] for tup in lsa_model[dictionary.doc2bow(sentence)]])
    lsa_vectors.append(lsa_vector)

# Print the vector representation of the preprocessed sentences
print('LSA Vectors of the model:')
for i, vector in enumerate(lsa_vectors):
    print(f'Sentence {i+1}: {vector.round(3)}')
    
# Get the top 3 topics based on the LSA model
lsa_topics = lsa_model.print_topics(num_topics=3)
print('\nTop 3 Topics of the model :')
for topic in lsa_topics:
    print(topic)
>>>>>>> origin/main
