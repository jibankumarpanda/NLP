paragraph "PM Narendra Modi is a 2019 Indian Hindi-language propaganda film.[5][6] It was directed by Omung Kumar, written by Anirudh Chawla and Vivek Oberoi, and produced under the banner of Legend Studios. The film is a hagiography of Narendra Modi, the prime minister of India since 2014.[7][8] Its original release schedule on the opening day of the 2019 Indian general election caused significant backlash, leading to the film being banned by the Election Commission of India (ECI) for the duration of the general election. It was only released after the election's conclusion, receiving overwhelmingly negative reviews by critics."
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

words = word_tokenize(paragraph)
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word not in stop_words]
print(filtered_words)