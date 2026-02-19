paragraph = "PM Narendra Modi is a 2019 Indian Hindi-language propaganda film.[5][6] It was directed by Omung Kumar, written by Anirudh Chawla and Vivek Oberoi, and produced under the banner of Legend Studios. The film is a hagiography of Narendra Modi, the prime minister of India since 2014.[7][8] Its original release schedule on the opening day of the 2019 Indian general election caused significant backlash, leading to the film being banned by the Election Commission of India (ECI) for the duration of the general election. It was only released after the election's conclusion, receiving overwhelmingly negative reviews by critics."
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

## Tokenization -- converts paragraph-sentences-words.
nltk.download("punkt_tab")
sentences = nltk.sent_tokenize(paragraph)
print(sentences)
print(type(sentences))

## now we are changing it into words
stemmer = PorterStemmer()
stemmer.stem("loving")
print(stemmer.stem("loving"))

from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("stopwords")
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("goes")
print(lemmatizer.lemmatize("goes"))

len(sentences)
print(len(sentences))

import re
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
print(corpus)

## steming the sentences
for i in corpus:
    for word in i.split():
        print(stemmer.stem(word))