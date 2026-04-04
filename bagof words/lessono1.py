paragraph = "PM Narendra Modi is a 2019 Indian Hindi-language propaganda film.[5][6] It was directed by Omung Kumar, written by Anirudh Chawla and Vivek Oberoi, and produced under the banner of Legend Studios. The film is a hagiography of Narendra Modi, the prime minister of India since 2014.[7][8] Its original release schedule on the opening day of the 2019 Indian general election caused significant backlash, leading to the film being banned by the Election Commission of India (ECI) for the duration of the general election. It was only released after the election's conclusion, receiving overwhelmingly negative reviews by critics."
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")

# Text preprocessing
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    return ' '.join(words)

# Preprocess the paragraph
processed_paragraph = preprocess_text(paragraph)
print("Processed paragraph:")
print(processed_paragraph)
print()

# Create sentences for bigram/trigram analysis
sentences = nltk.sent_tokenize(paragraph)
processed_sentences = [preprocess_text(sent) for sent in sentences]

print("Processed sentences:")
for i, sent in enumerate(processed_sentences):
    print(f"Sentence {i+1}: {sent}")
print()

# 1. Unigram (Bag of Words)
print("=== UNIGRAM ANALYSIS ===")
cv_unigram = CountVectorizer(ngram_range=(1,1))
X_unigram = cv_unigram.fit_transform(processed_sentences)

print("Unigram Vocabulary:")
print(cv_unigram.vocabulary_)
print()

print("Unigram Feature Names:")
print(cv_unigram.get_feature_names_out())
print()

print("Unigram Matrix Shape:", X_unigram.shape)
print("Unigram Matrix:")
print(X_unigram.toarray())
print()

# 2. Bigram Analysis
print("=== BIGRAM ANALYSIS ===")
cv_bigram = CountVectorizer(ngram_range=(2,2))
X_bigram = cv_bigram.fit_transform(processed_sentences)

print("Bigram Vocabulary:")
print(cv_bigram.vocabulary_)
print()

print("Bigram Feature Names:")
print(cv_bigram.get_feature_names_out())
print()

print("Bigram Matrix Shape:", X_bigram.shape)
print("Bigram Matrix:")
print(X_bigram.toarray())
print()

# 3. Trigram Analysis
print("=== TRIGRAM ANALYSIS ===")
cv_trigram = CountVectorizer(ngram_range=(3,3))
X_trigram = cv_trigram.fit_transform(processed_sentences)

print("Trigram Vocabulary:")
print(cv_trigram.vocabulary_)
print()

print("Trigram Feature Names:")
print(cv_trigram.get_feature_names_out())
print()

print("Trigram Matrix Shape:", X_trigram.shape)
print("Trigram Matrix:")
print(X_trigram.toarray())
print()

# 4. Combined N-grams (1-3 grams)
print("=== COMBINED N-GRAM ANALYSIS (1-3 grams) ===")
cv_combined = CountVectorizer(ngram_range=(1,3))
X_combined = cv_combined.fit_transform(processed_sentences)

print("Combined Vocabulary Size:", len(cv_combined.vocabulary_))
print("Combined Vocabulary:")
print(cv_combined.vocabulary_)
print()

print("Combined Feature Names:")
print(cv_combined.get_feature_names_out())
print()

print("Combined Matrix Shape:", X_combined.shape)
print("Combined Matrix:")
print(X_combined.toarray())
print()

# 5. Sparsity Analysis
print("=== SPARSITY ANALYSIS ===")
def calculate_sparsity(matrix):
    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    sparsity = 1 - density
    return sparsity

print("Unigram Sparsity:", calculate_sparsity(X_unigram))
print("Bigram Sparsity:", calculate_sparsity(X_bigram))
print("Trigram Sparsity:", calculate_sparsity(X_trigram))
print("Combined Sparsity:", calculate_sparsity(X_combined))
print()

# 6. X[0].toarray() demonstration
print("=== X[0].toarray() DEMONSTRATION ===")
print("First document (unigram):", X_unigram[0].toarray())
print("First document (bigram):", X_bigram[0].toarray())
print("First document (trigram):", X_trigram[0].toarray())
print("First document (combined):", X_combined[0].toarray())
print()

# 7. TF-IDF Analysis for comparison
print("=== TF-IDF ANALYSIS ===")
tfidf = TfidfVectorizer(ngram_range=(1,1))
X_tfidf = tfidf.fit_transform(processed_sentences)

print("TF-IDF Vocabulary:")
print(tfidf.vocabulary_)
print()

print("TF-IDF Matrix Shape:", X_tfidf.shape)
print("TF-IDF Matrix:")
print(X_tfidf.toarray())
print()

print("TF-IDF Sparsity:", calculate_sparsity(X_tfidf))