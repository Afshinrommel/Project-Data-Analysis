

# ===============================================================
# 📦 1. IMPORT DEPENDENCIES
# ---------------------------------------------------------------
# Imports all the required libraries for NLP, topic modeling,
# coherence evaluation and visualization. (Use only once in Google Colab)
# ===============================================================

# Installations (Colab cell magic allowed)

import warnings
import nltk
import pyLDAvis
import contractions
import pandas as pd
import re
import matplotlib.pyplot as plt
import unicodedata
from bs4 import BeautifulSoup
import spacy
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
#from google.colab import drive
import opendatasets as od
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from bertopic import BERTopic

# For coherence
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from tqdm import tqdm

# ===============================================================
# ⚙️ 2. ENVIRONMENT SETUP
# ---------------------------------------------------------------
# Downloads NLTK stopwords and lemmatization data,
# loads SpaCy model for NLP tasks,
# and mounts Google Drive (for file access in Colab).
# ===============================================================
warnings.filterwarnings("ignore", category=DeprecationWarning)
nltk.download("stopwords")
nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm")

# Mount Google Drive (only required in Google Colab)
#drive.mount('/content/drive')

# ===============================================================
# 📥 3. LOAD & EXPLORE DATA
# ---------------------------------------------------------------
# Downloads the TripAdvisor Hotel Reviews dataset from Kaggle
# and provides an initial look at review ratings distribution.
# ===============================================================
#od.download("https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews")

# Load dataset
raw_data = pd.read_csv('C:/Users/Dell/Documents/semester 5/data_analysis_project/on windows/tripadvisor_hotel_reviews.csv')
raw_data['Review'] = raw_data['Review'].str.replace('*', ' stars')

# Visualize distribution of ratings
fig = px.histogram(
    raw_data, x='Rating', title='Histogram of Review Ratings',
    template='plotly_dark', color='Rating',
    color_discrete_sequence=px.colors.sequential.Blues_r,
    opacity=0.8, height=525, width=835
)
fig.update_yaxes(title='Count')
fig.show()

# Copy original data for processing
df = raw_data.copy()

# ===============================================================
# 🧹 4. TEXT CLEANING & PREPROCESSING
# ---------------------------------------------------------------
# Performs comprehensive text cleaning:
# - Remove emojis, HTML tags, URLs, and special symbols
# - Convert text to lowercase
# - Expand contractions like “can’t” → “cannot”
# - Remove punctuation, numbers, and extra spaces
# ===============================================================

def remove_emoji(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" "+]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Some rows may be NaN — guard against that
df['Review'] = df['Review'].fillna('').astype(str)

df["Review"] = df["Review"].apply(remove_emoji)


def remove_html_tags(text):
    return BeautifulSoup(text, 'html.parser').get_text()

df["Review"] = df["Review"].apply(remove_html_tags)

df["Review"] = df["Review"].str.lower()

def remove_url(text):
    return re.sub(r'https?:\S*', '', text)

def remove_punctuation(text):
    import string
    return ''.join([c for c in text if c not in string.punctuation])

df["Review"] = df["Review"].replace(r'\s+', ' ', regex=True)
df["Review"] = df["Review"].str.replace('<br />', '')

# ===============================================================
# 🧽 5. TEXT NORMALIZATION, STOPWORD REMOVAL & LEMMATIZATION
# ---------------------------------------------------------------
# - Removes small and numeric tokens
# - Filters stopwords using NLTK list
# - Lemmatizes tokens using SpaCy (e.g., “running” → “run”)
# ===============================================================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = contractions.fix(text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\b\w{1,2}\b", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    return " ".join([w for w in text.split() if w.lower() not in stop_words])


def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Apply cleaning pipeline
df["Review"] = df["Review"].apply(clean_text).apply(remove_stopwords).apply(lemmatize)
print(df.head())

# WordCloud analyzes word frequency and creates a visual.
text = " ".join(df['Review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(8, 4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# ===============================================================

# ===============================================================
# 🔠 6. FEATURE EXTRACTION (TF-IDF VECTORIZATION)
# ---------------------------------------------------------------
# Converts text data into a numerical form using TF-IDF (Term Frequency–Inverse Document Frequency).
# This representation is ideal for algorithms like NMF and SVD
# because it emphasizes unique, informative words while reducing common ones.
# ===============================================================

tfidf_vectorizer = TfidfVectorizer(stop_words=list(spacy_stopwords), min_df=5, max_df=0.7)
tfidf_vectors = tfidf_vectorizer.fit_transform(df["Review"])

# Also prepare CountVectorizer for LDA
count_vectorizer = CountVectorizer(stop_words=list(spacy_stopwords), min_df=5, max_df=0.7)
count_vectors = count_vectorizer.fit_transform(df["Review"])

# ===============================================================
# ⚖️ COHERENCE SCORE SETUP
# ---------------------------------------------------------------
# We'll use Gensim's CoherenceModel to evaluate topic coherence.
# Prepare corpus and dictionary for coherence computation
# ===============================================================
texts = [doc.split() for doc in df["Review"]]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Helper: display topics (keeps your original function)
def display_topics(h, features, no_top_words=5):
    for topic, words in enumerate(h):
        largest = words.argsort()[::-1]
        print(f"\nTopic {topic}")
        for i in range(no_top_words):
            print("  %s" % (features[largest[i]]))

# ===============================================================
# 🧩 7. TOPIC MODELING - NMF (Non-negative Matrix Factorization)
# ---------------------------------------------------------------
# Keep original NMF flow and add coherence-based exploration
# ===============================================================
nmf = NMF(n_components=10, random_state=42, max_iter=600)
W_nmf = nmf.fit_transform(tfidf_vectors)
H_nmf = nmf.components_

display_topics(H_nmf, tfidf_vectorizer.get_feature_names_out())

# --- Coherence sweep for NMF ---

def compute_coherence_nmf(start=5, stop=50, step=1):
    coherence_scores = []
    topic_range = list(range(start, stop + 1, step))
    for n_topics in tqdm(topic_range, desc='NMF coherence'):
        model_tmp = NMF(n_components=n_topics, random_state=42, max_iter=600)
        W = model_tmp.fit_transform(tfidf_vectors)
        H = model_tmp.components_
        topics = [
            [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
            for topic in H
        ]
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(cm.get_coherence())
    plt.plot(topic_range, coherence_scores, marker='o')
    plt.title("NMF Coherence Score by Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.show()
    return pd.DataFrame({'n_topics': topic_range, 'coherence': coherence_scores})

nmf_scores_df = compute_coherence_nmf(5, 50, 1)
print(nmf_scores_df.sort_values('coherence', ascending=False).head())

# ===============================================================
# 🧩 8. TOPIC MODELING - SVD (Latent Semantic Analysis / LSA)
# ---------------------------------------------------------------
# Keep original SVD flow and add coherence-based exploration
# ===============================================================
svd = TruncatedSVD(n_components=10, random_state=42)
W_svd = svd.fit_transform(tfidf_vectors)
H_svd = svd.components_
display_topics(H_svd, tfidf_vectorizer.get_feature_names_out())

# --- Coherence sweep for SVD/LSA ---

def compute_coherence_svd(start=5, stop=50, step=1):
    coherence_scores = []
    topic_range = list(range(start, stop + 1, step))
    for n_topics in tqdm(topic_range, desc='SVD coherence'):
        svd_tmp = TruncatedSVD(n_components=n_topics, random_state=42)
        W = svd_tmp.fit_transform(tfidf_vectors)
        H = svd_tmp.components_
        topics = [
            [tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
            for topic in H
        ]
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(cm.get_coherence())
    plt.plot(topic_range, coherence_scores, marker='o', color='green')
    plt.title("LSA/SVD Coherence Score by Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.show()
    return pd.DataFrame({'n_topics': topic_range, 'coherence': coherence_scores})

svd_scores_df = compute_coherence_svd(5, 50, 1)
print(svd_scores_df.sort_values('coherence', ascending=False).head())

# ===============================================================
# 🧩 9. TOPIC MODELING - LDA (Latent Dirichlet Allocation)
# ---------------------------------------------------------------
# Keep original LDA flow and add coherence-based exploration
# ===============================================================
lda = LatentDirichletAllocation(n_components=10, random_state=42, n_jobs=4)
lda.fit(count_vectors)

pyLDAvis.enable_notebook()
vis = pyLDAvis.prepare(
    topic_term_dists=lda.components_ / lda.components_.sum(axis=1)[:, None],
    doc_topic_dists=lda.transform(count_vectors),
    doc_lengths=count_vectors.sum(axis=1).A1,
    vocab=count_vectorizer.get_feature_names_out(),
    term_frequency=count_vectors.sum(axis=0).A1
)

# If running in Jupyter/Colab this will render the visualization
vis

# --- Coherence sweep for LDA ---

def compute_coherence_lda(start=5, stop=50, step=1):
    coherence_scores = []
    topic_range = list(range(start, stop + 1, step))
    for n_topics in tqdm(topic_range, desc='LDA coherence'):
        lda_tmp = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
        lda_tmp.fit(count_vectors)
        topics = [
            [count_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
            for topic in lda_tmp.components_
        ]
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(cm.get_coherence())
    plt.plot(topic_range, coherence_scores, marker='o', color='orange')
    plt.title("LDA Coherence Score by Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.show()
    return pd.DataFrame({'n_topics': topic_range, 'coherence': coherence_scores})

lda_scores_df = compute_coherence_lda(5, 50, 1)
print(lda_scores_df.sort_values('coherence', ascending=False).head())

# ===============================================================
# 🧠 10. TOPIC MODELING - BERTopic
# ---------------------------------------------------------------
# Keep original BERTopic flow and compute coherence for the found topics
# ===============================================================
model = BERTopic(verbose=True, embedding_model="paraphrase-MiniLM-L3-v2", min_topic_size=7)
topics, _ = model.fit_transform(df["Review"])

freq = model.get_topic_info()
print("Number of topics:", len(freq))
print(freq.head())

# Display words of a sample topic
if len(freq) > 1:
    a_topic = freq.iloc[1]["Topic"]
    print(model.get_topic(a_topic))

# Visualize topic distribution (keeps your existing visualization)
# model.visualize_barchart(top_n_topics=30)
model.visualize_barchart(top_n_topics=25)

# --- Coherence for BERTopic topics ---
# Extract topic word lists, ignoring outlier topic -1
topic_words = []
for topic_id in freq['Topic'].tolist():
    if topic_id == -1:
        continue
    topic = model.get_topic(topic_id)
    if topic is None:
        continue
    words = [w for w, _ in topic]
    topic_words.append(words)

if topic_words:
    cm = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
    bertopic_coherence = cm.get_coherence()
    print(f"BERTopic Coherence Score (c_v): {bertopic_coherence:.4f}")
else:
    print("No BERTopic topics available for coherence computation")

# ===============================================================
# 🏁 Interpretation & Suggestions
# ---------------------------------------------------------------
# Each of the *_scores_df DataFrames contains coherence scores for the
# range of topics tested. The highest coherence value indicates the
# recommended number of topics (by c_v coherence) for that model.
# Example:
# nmf_scores_df.sort_values('coherence', ascending=False).head(3)
# lda_scores_df.sort_values('coherence', ascending=False).head(3)
# svd_scores_df.sort_values('coherence', ascending=False).head(3)
# ===============================================================

print('\nTop coherence results:')
print('NMF best:\n', nmf_scores_df.sort_values('coherence', ascending=False).head(3))
print('\nLSA/SVD best:\n', svd_scores_df.sort_values('coherence', ascending=False).head(3))
print('\nLDA best:\n', lda_scores_df.sort_values('coherence', ascending=False).head(3))

# End of integrated notebook

# ===============================================================
# 🧠 10. TOPIC MODELING - BERTopic with Coherence Evaluation
# ---------------------------------------------------------------
# We'll evaluate how coherence changes for different BERTopic
# configurations by varying `min_topic_size`.
# ===============================================================
from bertopic import BERTopic
from gensim.models import CoherenceModel
from tqdm import tqdm

# Define different topic size thresholds to test
min_topic_sizes = [5, 7, 10, 15, 20, 30, 50]
bertopic_scores = []

for size in tqdm(min_topic_sizes, desc="BERTopic coherence sweep"):
    # Build and fit model
    model_tmp = BERTopic(
        verbose=False,
        embedding_model="paraphrase-MiniLM-L3-v2",
        min_topic_size=size
    )
    topics_tmp, _ = model_tmp.fit_transform(df["Review"])

    # Extract topic words
    freq_tmp = model_tmp.get_topic_info()
    topic_words = []
    for topic_id in freq_tmp["Topic"]:
        if topic_id == -1:
            continue
        words = [w for w, _ in model_tmp.get_topic(topic_id)]
        topic_words.append(words)

    # Compute coherence if topics are found
    if topic_words:
        cm = CoherenceModel(
            topics=topic_words,
            texts=texts,
            dictionary=dictionary,
            coherence="c_v"
        )
        coherence = cm.get_coherence()
        bertopic_scores.append(coherence)
    else:
        bertopic_scores.append(0)

# 📈 Plot coherence vs. min_topic_size
plt.figure(figsize=(8, 5))
plt.plot(min_topic_sizes, bertopic_scores, marker='o', color='purple')
plt.title("BERTopic Coherence Score by min_topic_size")
plt.xlabel("min_topic_size")
plt.ylabel("Coherence Score (c_v)")
plt.grid(True)
plt.show()

# 🏁 Display the best configuration
best_idx = int(np.argmax(bertopic_scores))
best_size = min_topic_sizes[best_idx]
best_score = bertopic_scores[best_idx]
print(f"✅ Best BERTopic coherence: {best_score:.4f} (min_topic_size={best_size})")

# 🔁 Train a final BERTopic model using the best parameter
model_best = BERTopic(
    verbose=True,
    embedding_model="paraphrase-MiniLM-L3-v2",
    min_topic_size=best_size
)
topics_best, _ = model_best.fit_transform(df["Review"])

# Visualize and inspect
freq_best = model_best.get_topic_info()

print(f"Number of topics (best): {len(freq_best)}")

model_best.visualize_barchart(top_n_topics=25)

# ===============================================================
# 🧠 10. TOPIC MODELING - BERTopic with Coherence Evaluation (2–50)
# ---------------------------------------------------------------
# We'll evaluate how coherence changes for different BERTopic
# configurations by varying `min_topic_size` from 2 to 50.
# ===============================================================
from bertopic import BERTopic
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from tqdm import tqdm
import matplotlib.pyplot as plt

# 🧹 Prepare tokenized texts and dictionary
texts = [str(doc).split() for doc in df["Review"]]  # simple whitespace tokenization
dictionary = Dictionary(texts)

# Define range of valid min_topic_size values (2–50)
min_topic_sizes = list(range(2, 51))
bertopic_scores = []

for size in tqdm(min_topic_sizes, desc="BERTopic coherence sweep (2–50)"):
    try:
        # Build and fit model
        model_tmp = BERTopic(
            verbose=False,
            embedding_model="paraphrase-MiniLM-L3-v2",
            min_topic_size=size
        )
        topics_tmp, _ = model_tmp.fit_transform(df["Review"])

        # Extract topic words
        freq_tmp = model_tmp.get_topic_info()
        topic_words = []
        for topic_id in freq_tmp["Topic"]:
            if topic_id == -1:
                continue
            words = [w for w, _ in model_tmp.get_topic(topic_id)]
            topic_words.append(words)

        # Compute coherence if topics are found
        if topic_words:
            cm = CoherenceModel(
                topics=topic_words,
                texts=texts,
                dictionary=dictionary,
                coherence="c_v"
            )
            coherence = cm.get_coherence()
            bertopic_scores.append(coherence)
        else:
            bertopic_scores.append(0)
    except Exception as e:
        print(f"⚠️ Skipped min_topic_size={size} due to error: {e}")
        bertopic_scores.append(0)

# 📈 Plot coherence vs. min_topic_size
plt.figure(figsize=(8, 5))
plt.plot(min_topic_sizes, bertopic_scores, marker='o', color='purple')
plt.title("BERTopic Coherence Score by min_topic_size (2–50)")
plt.xlabel("min_topic_size")
plt.ylabel("Coherence Score (c_v)")
plt.grid(True)
plt.show()

# 🧾 Print best configuration
best_size = min_topic_sizes[bertopic_scores.index(max(bertopic_scores))]
print(f"\n🏆 Best min_topic_size = {best_size} with coherence = {max(bertopic_scores):.4f}")

# 🏁 Display the best configuration
best_idx = int(np.argmax(bertopic_scores))
best_size = min_topic_sizes[best_idx]
best_score = bertopic_scores[best_idx]
print(f"✅ Best BERTopic coherence: {best_score:.4f} (min_topic_size={best_size})")

# 🔁 Train a final BERTopic model using the best parameter
model_best = BERTopic(
    verbose=True,
    embedding_model="paraphrase-MiniLM-L3-v2",
    min_topic_size=best_size
)
topics_best, _ = model_best.fit_transform(df["Review"])

# Visualize and inspect
freq_best = model_best.get_topic_info()

model_best.visualize_barchart(top_n_topics=14)