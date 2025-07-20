import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np
import spacy
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import os

# --- Configura i Parametri Qui ---
INPUT_FILE_PATH = 'emotions_dataset.parquet'  # Assicurati che il tuo file sia qui o specifica il percorso completo
SENTENCE_COLUMN = 'Sentence'
LABEL_COLUMN = 'Label' # Necessario per caricare il dataset, anche se non usato nel task specifico
MAX_DF = 0.8  # Ignora parole che appaiono >80% dei documenti
MIN_DF = 7  # Ignora parole che appaiono meno di 7 volte
SPACY_MODEL = "en_core_web_sm"  # Usa "it_core_news_sm" se il tuo testo è in italiano
N_TOP_WORDS_FOR_COHERENCE = 30 # Numero di parole top da considerare per il calcolo della coerenza (parametro 'topn' di CoherenceModel)

# --- Parametri per la ricerca del numero ottimale di topic ---
MIN_TOPICS = 2
MAX_TOPICS = 20 # Scegli un range adeguato per i tuoi dati
STEP_TOPICS = 1

# --- Carica il Modello spaCy ---
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    print(f"Modello spaCy '{SPACY_MODEL}' non trovato. Esegui: python -m spacy download {SPACY_MODEL}")
    print("Assicurati di aver scaricato il modello corretto per la lingua del tuo dataset.")
    exit()

# --- Funzione per la Lemmatizzazione del Testo ---
def lemmatize_text(text):
    """Lemmatizza una stringa di testo, rimuovendo stopword e punteggiatura."""
    doc = nlp(text)
    # Filtra token che non sono stopword, punteggiatura, spazi, numeri o non-alfabetiche
    lemmas = [token.lemma_ for token in doc if
              not token.is_stop and not token.is_punct and not token.is_space and not token.like_num and token.is_alpha]
    return " ".join(lemmas)

# --- Caricamento e Preparazione del Dataset ---
try:
    df = pd.read_parquet(INPUT_FILE_PATH)
except FileNotFoundError:
    print(f"Errore: Il file '{INPUT_FILE_PATH}' non è stato trovato.")
    exit()

df_cleaned = df.dropna(subset=[SENTENCE_COLUMN, LABEL_COLUMN]).copy()
# Applica lemmatizzazione. Converti in stringa per gestire potenziali non-stringhe prima di spaCy
df_cleaned[SENTENCE_COLUMN] = df_cleaned[SENTENCE_COLUMN].astype(str).apply(lemmatize_text)

# Rimuovi righe dove la lemmatizzazione ha prodotto una stringa vuota
df_cleaned = df_cleaned[df_cleaned[SENTENCE_COLUMN].str.strip() != ''].copy()

sentences = df_cleaned[SENTENCE_COLUMN].tolist()

if not sentences:
    print("Nessuna frase valida trovata dopo la pulizia e lemmatizzazione.")
    exit()

# --- Vettorizzazione del Testo ---
vectorizer = CountVectorizer(max_df=MAX_DF, min_df=MIN_DF)
dtm = vectorizer.fit_transform(sentences)
feature_names = vectorizer.get_feature_names_out()

# Per la coerenza Gensim, abbiamo bisogno del corpus tokenizzato e del dizionario
tokenized_texts = [text.split() for text in sentences]
dictionary = Dictionary(tokenized_texts)
# Il corpus Gensim non è strettamente necessario per la coerenza c_v con `texts`
# ma lo teniamo per completezza e se si volesse usare un altro tipo di coerenza
corpus_gensim = [dictionary.doc2bow(text) for text in tokenized_texts]


# --- Funzione per calcolare la coerenza ---
def compute_coherence_values(dictionary, texts, dtm, feature_names, min_topics, max_topics, step, topn_words):
    coherence_scores = []
    # model_list = [] # Non necessario per questo task
    for num_topics in range(min_topics, max_topics + 1, step):
        print(f"Calcolo coerenza per {num_topics} topics...")
        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_model.fit(dtm)
        # model_list.append(lda_model) # Non necessario

        # Estrazione delle top words per il calcolo della coerenza (metodo comune per sklearn con Gensim)
        topic_words = []
        for topic_comp in lda_model.components_:
            # Ordina gli indici delle parole per probabilità decrescente
            top_word_indices = topic_comp.argsort()[-topn_words:][::-1]
            # Mappa gli indici alle parole reali
            topic_words.append([feature_names[i] for i in top_word_indices])

        coherence_model = CoherenceModel(topics=topic_words, texts=texts,
                                         dictionary=dictionary, coherence='c_v')
        coherence_scores.append(coherence_model.get_coherence())

    return coherence_scores

# --- Trova il numero ottimale di topic ---
coherence_values = compute_coherence_values(
    dictionary=dictionary,
    texts=tokenized_texts,
    dtm=dtm,
    feature_names=feature_names,
    min_topics=MIN_TOPICS,
    max_topics=MAX_TOPICS,
    step=STEP_TOPICS,
    topn_words=N_TOP_WORDS_FOR_COHERENCE
)

# Plot della coerenza per il numero di topic
x = range(MIN_TOPICS, MAX_TOPICS + 1, STEP_TOPICS)
plt.figure(figsize=(10, 6))
plt.plot(x, coherence_values)
plt.xlabel("Numero di Topics")
plt.ylabel("Coerenza del Modello (c_v)")
plt.title("Coerenza del Modello LDA per Numero di Topics")
plt.grid(True)
plt.xticks(x)
plt.savefig('lda_coherence_scores.png')
print("\nGrafico della coerenza salvato come 'lda_coherence_scores.png'.")
plt.close()

# Trova il numero di topic con la coerenza massima
optimal_topics_index = np.argmax(coherence_values)
N_TOPICS_OPTIMAL = x[optimal_topics_index]
optimal_coherence_score = coherence_values[optimal_topics_index]

print(f"\nIl numero ottimale di topics (basato sulla coerenza c_v) è: {N_TOPICS_OPTIMAL}")
print(f"Punteggio di coerenza corrispondente: {optimal_coherence_score:.4f}")

print("\nProcesso di identificazione del numero ottimale di topics completato.")
print("Controlla il file 'lda_coherence_scores.png' per visualizzare il grafico della coerenza.")