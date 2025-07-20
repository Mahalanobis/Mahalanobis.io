import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.lda_model
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import spacy
from wordcloud import WordCloud
import os
import glob
import imageio as imageio

# --- Configura i Parametri Qui ---
INPUT_FILE_PATH = 'emotions_dataset.parquet'
SENTENCE_COLUMN = 'Sentence'
LABEL_COLUMN = 'Label'
N_TOPICS = 10 # Numero di topic per l'LDA
MAX_DF = 0.8 # Frequenza massima del documento per le parole (ignora parole troppo comuni)
MIN_DF = 7 # Frequenza minima del documento per le parole (ignora parole troppo rare)
SPACY_MODEL = "en_core_web_sm" # Modello spaCy da usare (es. "en_core_web_sm" per inglese, "it_core_news_sm" per italiano)
N_TOP_WORDS_WORDCLOUD = 100 # Numero di parole da mostrare in ciascuna word cloud
N_TOP_WORDS_LDA_TEXT_OUTPUT = 30 # Numero di parole da mostrare per topic nel file di testo LDA

# --- Configura Parametri GIF ---
GIF_OUTPUT_FILENAME = 'emotion_wordclouds_animation.gif'
GIF_FRAME_DURATION = 500 # Durata di ogni frame in millisecondi

# --- Carica il Modello spaCy ---
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    print(f"Modello spaCy '{SPACY_MODEL}' non trovato. Esegui: python -m spacy download {SPACY_MODEL}")
    print("Assicurati di aver scaricato il modello corretto per la lingua del tuo dataset.")
    exit()

# --- Funzione per la Lemmatizzazione del Testo ---
def lemmatize_text(text):
    """
    Applica la lemmatizzazione al testo, rimuovendo stop words, punteggiatura, spazi, numeri
    e mantenendo solo le parole alfabetiche.
    """
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if
              not token.is_stop and not token.is_punct and not token.is_space and not token.like_num and token.is_alpha]
    return " ".join(lemmas)

# --- Caricamento e Preparazione del Dataset ---
try:
    df = pd.read_parquet(INPUT_FILE_PATH)
    print(f"Dataset '{INPUT_FILE_PATH}' caricato con successo. Righe iniziali: {len(df)}")
except FileNotFoundError:
    print(f"Errore: Il file '{INPUT_FILE_PATH}' non è stato trovato. Assicurati che sia nella stessa directory dello script o che il percorso sia corretto.")
    exit()

# Pulizia e lemmatizzazione delle frasi
df_cleaned = df.dropna(subset=[SENTENCE_COLUMN, LABEL_COLUMN]).copy()
df_cleaned[SENTENCE_COLUMN] = df_cleaned[SENTENCE_COLUMN].astype(str).apply(lemmatize_text)
df_cleaned = df_cleaned[df_cleaned[SENTENCE_COLUMN].str.strip() != ''].copy() # Rimuovi righe con frasi vuote dopo lemmatizzazione

sentences = df_cleaned[SENTENCE_COLUMN].tolist()
labels = df_cleaned[LABEL_COLUMN].tolist()

if not sentences:
    print("Nessuna frase valida trovata dopo la pulizia e lemmatizzazione. Controlla il dataset e i parametri.")
    exit()
else:
    print(f"Numero di frasi valide dopo la pulizia e lemmatizzazione: {len(sentences)}")

# --- Vettorizzazione del Testo per LDA (con filtering min_df/max_df) ---
print(f"\nVettorizzazione del testo con CountVectorizer (max_df={MAX_DF}, min_df={MIN_DF})...")
vectorizer = CountVectorizer(max_df=MAX_DF, min_df=MIN_DF)
dtm = vectorizer.fit_transform(sentences)
feature_names = vectorizer.get_feature_names_out()
print(f"Matrice DTM creata. Vocabolario di {len(feature_names)} termini.")

# --- Addestramento del Modello LDA ---
print(f"\nAddestramento del Modello LDA con {N_TOPICS} topic...")
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42)
lda.fit(dtm)
print("Addestramento LDA completato.")

# --- Visualizzazione Interattiva dei Topic con pyLDAvis ---
print("\nPreparazione della visualizzazione interattiva dei topic con pyLDAvis...")
vis = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)
pyLDAvis.save_html(vis, 'lda_topics_visualization.html')
print("Visualizzazione LDA salvata come 'lda_topics_visualization.html'.")

# --- NEW ROBUST TOPIC MAPPING ---
print("\n--- Generating PyLDAvis-aligned topic names and order ---")
# The 'topic_info' DataFrame in 'vis' contains the information for *each word* across all topics.
# The 'Category' column tells us which pyLDAvis displayed topic ('Topic1', 'Topic2', etc.) a row belongs to.
# We need to iterate through these categories in their natural display order (Topic1, Topic2, ...)
# and for each category, identify which of our original LDA model's topics (0, 1, ...) it corresponds to.

# First, let's collect the top words for each *displayed* pyLDAvis topic.
pyldavis_displayed_topics_data = {}
if hasattr(vis, 'topic_info') and isinstance(vis.topic_info, pd.DataFrame):
    # Get unique 'TopicN' labels, excluding 'Default' (which represents overall terms)
    pyldavis_categories = sorted([c for c in vis.topic_info['Category'].unique() if c.startswith('Topic')],
                                 key=lambda x: int(x.replace('Topic', '')))

    for category_name in pyldavis_categories:
        # Filter topic_info for the current category and sort by logprob (word relevance within that topic)
        top_words_df = vis.topic_info[vis.topic_info['Category'] == category_name].sort_values(by='logprob', ascending=False)
        pyldavis_displayed_topics_data[category_name] = {
            'words': top_words_df['Term'].head(N_TOP_WORDS_LDA_TEXT_OUTPUT).tolist(),
            'freq': top_words_df['Freq'].iloc[0] if not top_words_df.empty else 0 # Freq of the top word for this topic
        }
else:
    print("Warning: vis.topic_info not available or not a DataFrame. Cannot precisely map pyLDAvis topics.")
    print("Falling back to sorting by original LDA topic frequency (less accurate mapping to pyLDAvis display).")
    # This is the fallback if vis.topic_info is missing, similar to your old topic_freq approach
    # In this scenario, we cannot guarantee exact pyLDAvis alignment.
    ordered_original_indices = np.argsort(-vis.topic_freq).tolist() if hasattr(vis, 'topic_freq') else list(range(N_TOPICS))
    pyldavis_names_map = {original_idx: f"Topic {i + 1}" for i, original_idx in enumerate(ordered_original_indices)}


# Now, let's match these pyLDAvis displayed topics to the original LDA model's topics (0 to N-1)
# We'll build a mapping from pyLDAvis displayed name ('Topic1') to original LDA index (e.g., 7)
pyldavis_to_original_lda_map = {}
original_lda_to_pyldavis_name_map = {} # This will be our `pyldavis_names_map`

# Create a list of top words for each original LDA topic for comparison
original_lda_top_words = {}
for original_lda_idx in range(N_TOPICS):
    topic_data = lda.components_[original_lda_idx]
    top_words_indices = topic_data.argsort()[:-N_TOP_WORDS_LDA_TEXT_OUTPUT - 1:-1]
    original_lda_top_words[original_lda_idx] = [feature_names[i] for i in top_words_indices]

# Perform matching for each pyLDAvis displayed topic
unmatched_original_lda_topics = set(range(N_TOPICS))

for pyldavis_label in pyldavis_categories:
    pyldavis_words = pyldavis_displayed_topics_data[pyldavis_label]['words']
    best_match_original_idx = -1
    highest_similarity = -1

    # Only consider original LDA topics that haven't been matched yet
    for original_lda_idx in list(unmatched_original_lda_topics):
        current_original_lda_words = original_lda_top_words[original_lda_idx]

        # Calculate Jaccard similarity
        set_pyldavis = set(pyldavis_words)
        set_original_lda = set(current_original_lda_words)
        intersection = len(set_pyldavis.intersection(set_original_lda))
        union = len(set_pyldavis.union(set_original_lda))
        similarity = intersection / union if union > 0 else 0

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_original_idx = original_lda_idx

    # If a good match is found, assign it and remove from unmatched set
    if best_match_original_idx != -1 and highest_similarity > 0.1: # Threshold for reasonable match
        pyldavis_to_original_lda_map[pyldavis_label] = best_match_original_idx
        original_lda_to_pyldavis_name_map[best_match_original_idx] = pyldavis_label
        unmatched_original_lda_topics.remove(best_match_original_idx)
        print(f"PyLDAvis {pyldavis_label} matched to Original LDA Topic {best_match_original_idx} (Similarity: {highest_similarity:.2f})")
    else:
        # Fallback for topics that don't find a strong match (e.g., very general topics)
        # Or if we couldn't match all N_TOPICS, assign remaining original topics in numerical order
        print(f"Warning: PyLDAvis {pyldavis_label} could not find a strong match. Assigning numerically if unmatched original topics remain.")


# Finalize ordered_original_indices based on the pyLDAvis display order
ordered_original_indices = []
for label in pyldavis_categories:
    if label in pyldavis_to_original_lda_map:
        ordered_original_indices.append(pyldavis_to_original_lda_map[label])
    else:
        # If a pyLDAvis label somehow didn't map, fill it with an unmatched original topic
        if unmatched_original_lda_topics:
            next_unmatched = sorted(list(unmatched_original_lda_topics))[0]
            ordered_original_indices.append(next_unmatched)
            original_lda_to_pyldavis_name_map[next_unmatched] = label # Assign this label
            unmatched_original_lda_topics.remove(next_unmatched)
        else:
            print(f"Error: Ran out of original LDA topics to map for PyLDAvis label {label}. This should not happen if N_TOPICS matches.")


# Ensure all original topics are in the map, assign default names for any unmapped if necessary
# This handles cases where N_TOPICS might not perfectly align with pyLDAvis's perceived topics
for i in range(N_TOPICS):
    if i not in original_lda_to_pyldavis_name_map:
        original_lda_to_pyldavis_name_map[i] = f"Topic {i+1} (Unmapped)"
        if i not in ordered_original_indices:
             ordered_original_indices.append(i) # Ensure all original topics are considered

# Sort ordered_original_indices to reflect the desired output order (based on pyLDAvis_categories)
# The `ordered_original_indices` now directly corresponds to 'Topic1', 'Topic2', etc.
# We also need a reliable `pyldavis_names_map` that maps original LDA index to its 'TopicN' label.
pyldavis_names_map = original_lda_to_pyldavis_name_map

print("Final Mappatura topic (indice originale -> nome etichetta):", pyldavis_names_map)
print("Final Ordine dei topic per la heatmap e output (indici originali ordinati):", ordered_original_indices)


# --- Salva le parole chiave per ogni topic LDA in un file di testo ---
output_dir_lda_txt = 'lda_output'
os.makedirs(output_dir_lda_txt, exist_ok=True)
lda_topic_output_file = os.path.join(output_dir_lda_txt, 'lda_topics_top_words.txt')

with open(lda_topic_output_file, 'w', encoding='utf-8') as f:
    print(f"\nGenerazione del file '{lda_topic_output_file}' con le parole chiave dei topic LDA...")
    # Iterate through the topics in the order determined by pyLDAvis categories
    for original_topic_idx in ordered_original_indices:
        pyldavis_topic_name = pyldavis_names_map.get(original_topic_idx, f"Topic {original_topic_idx + 1} (Unmapped)")
        topic_data = lda.components_[original_topic_idx]

        top_words_indices = topic_data.argsort()[:-N_TOP_WORDS_LDA_TEXT_OUTPUT - 1:-1]
        top_words = [feature_names[i] for i in top_words_indices]
        top_words_weights = [topic_data[i] for i in top_words_indices]

        f.write(f"{pyldavis_topic_name} (Indice Originale: {original_topic_idx}):\n")
        for i, (word, weight) in enumerate(zip(top_words, top_words_weights)):
            f.write(f"  {i+1}. {word} ({weight:.3f})\n")
        f.write("---\n")
print(f"Parole chiave dei topic LDA salvate in '{lda_topic_output_file}'.")

# --- Analisi dell'Associazione Emozione-Topic con Odds Ratio ---
print("\nCalcolo delle distribuzioni di topic e associazione emozione-topic...")
topic_distributions = lda.transform(dtm)
df_cleaned['dominant_topic'] = topic_distributions.argmax(axis=1) # Assegna il topic dominante a ciascuna frase

contingency_table = pd.crosstab(df_cleaned[LABEL_COLUMN], df_cleaned['dominant_topic'])
odds_ratio_df = pd.DataFrame(index=contingency_table.index, columns=contingency_table.columns, dtype=float)

# Calcolo dell'Odds Ratio con correzione di Haldane-Anscombe
for emotion in contingency_table.index:
    for topic_idx in contingency_table.columns:
        a = contingency_table.loc[emotion, topic_idx] # Conteggio di (Emozione=True, Topic=True)
        b = contingency_table.loc[emotion, :].sum() - a # Conteggio di (Emozione=True, Topic=False)
        c = contingency_table.loc[:, topic_idx].sum() - a # Conteggio di (Emozione=False, Topic=True)
        d = contingency_table.sum().sum() - (a + b + c) # Conteggio di (Emozione=False, Topic=False)

        # Applica la correzione di Haldane-Anscombe per evitare divisioni per zero o logaritmi di zero
        a_prime = a + 0.5
        b_prime = b + 0.5
        c_prime = c + 0.5
        d_prime = d + 0.5

        if b_prime == 0 or d_prime == 0:
             odds_ratio = 0.0001 # Valore molto piccolo per evitare errori di logaritmo se uno dei denominatori è zero
        else:
            odds_ratio = (a_prime / b_prime) / (c_prime / d_prime)

        odds_ratio_df.loc[emotion, topic_idx] = odds_ratio

log_odds_ratio_df = np.log(odds_ratio_df)
print("Calcolo Log Odds Ratio completato.")

# Riordina e rinomina le colonne del DataFrame per la heatmap
# Use the `ordered_original_indices` and `pyldavis_names_map` derived from the new mapping logic
# Ensure that the columns selected are valid and in the correct order for the heatmap
log_odds_ratio_df_ordered = log_odds_ratio_df[ordered_original_indices]

# Rinomina le colonne con i nomi dei topic "PyLDAvis-like" (e.g., 'Topic1', 'Topic2')
log_odds_ratio_df_ordered.columns = [
    pyldavis_names_map.get(col, f"Topic {col + 1} (Name Missing)")
    for col in log_odds_ratio_df_ordered.columns
]

# --- Generazione della Heatmap ---
print("\nGenerazione della heatmap del Log Odds Ratio...")
plt.figure(figsize=(14, 9))
vmax = log_odds_ratio_df_ordered.abs().max().max()
sns.heatmap(log_odds_ratio_df_ordered, annot=True, fmt=".2f", cmap="RdBu", linewidths=.5, center=0, vmin=-vmax, vmax=vmax)
plt.title('Log Odds Ratio Emozione-LDA Topic')
plt.xlabel('LDA Topic')
plt.ylabel('Emozione (Label)')
plt.tight_layout()
plt.savefig('lda_emotion_topic_odds_ratio_heatmap_lemmatized.png')
print("Heatmap del Log Odds Ratio salvata come 'lda_emotion_topic_odds_ratio_heatmap_lemmatized.png'.")
plt.close()

# --- Generazione Word Clouds per Emozione basate su Log Odds Ratio ---
print("\nGenerazione delle Word Clouds per emozione...")
# Vettorizza tutto il vocabolario per calcolare gli odds ratio per parola
vectorizer_all_words = CountVectorizer()
dtm_all_words = vectorizer_all_words.fit_transform(df_cleaned[SENTENCE_COLUMN])
feature_names_all_words = vectorizer_all_words.get_feature_names_out()

# Crea maschere booleane per ogni etichetta di emozione
label_boolean_masks = {label: (df_cleaned[LABEL_COLUMN] == label).values for label in df_cleaned[LABEL_COLUMN].unique()}

all_words_list = list(feature_names_all_words)
total_docs = len(df_cleaned)

log_odds_ratios_words = {}
for label in df_cleaned[LABEL_COLUMN].unique():
    log_odds_ratios_words[label] = {}
    docs_with_label = label_boolean_masks[label].sum()
    docs_without_label = total_docs - docs_with_label

    for word_idx, word in enumerate(all_words_list):
        word_present_mask = (dtm_all_words[:, word_idx] > 0).toarray().flatten()

        a_word_in_label_docs = (word_present_mask & label_boolean_masks[label]).sum()
        b_no_word_in_label_docs = docs_with_label - a_word_in_label_docs
        c_word_not_in_label_docs = (word_present_mask & ~label_boolean_masks[label]).sum()
        d_no_word_not_in_label_docs = docs_without_label - c_word_not_in_label_docs

        a_prime = a_word_in_label_docs + 0.5
        b_prime = b_no_word_in_label_docs + 0.5
        c_prime = c_word_not_in_label_docs + 0.5
        d_prime = d_no_word_not_in_label_docs + 0.5

        if b_prime == 0 or d_prime == 0:
            odds_ratio = 0.0001
        else:
            odds_ratio = (a_prime / b_prime) / (c_prime / d_prime)

        if odds_ratio > 0:
            log_odds_ratios_words[label][word] = np.log(odds_ratio)
        else:
            log_odds_ratios_words[label][word] = 0.0

# Colormaps disponibili per le word clouds
cmap_list = [
    'viridis', 'plasma', 'magma', 'cividis', 'inferno',
    'Greens', 'Blues', 'Purples', 'Oranges', 'Reds',
    'bone', 'gist_heat', 'cool'
]
cmap_iterator = iter(cmap_list * ((len(df_cleaned[LABEL_COLUMN].unique()) // len(cmap_list)) + 1))

output_dir_wordclouds = 'wordclouds_by_emotion'
os.makedirs(output_dir_wordclouds, exist_ok=True)

for label, word_scores in log_odds_ratios_words.items():
    positive_log_odds_words = {word: score for word, score in word_scores.items() if score > 0}
    sorted_words = sorted(positive_log_odds_words.items(), key=lambda item: item[1], reverse=True)
    top_words_for_cloud = dict(sorted_words[:N_TOP_WORDS_WORDCLOUD])

    if not top_words_for_cloud:
        print(f"Nessuna parola significativa (con log odds ratio > 0) trovata per l'emozione: '{label}'.")
        continue

    current_cmap = next(cmap_iterator)
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=current_cmap).generate_from_frequencies(
        top_words_for_cloud)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Top {N_TOP_WORDS_WORDCLOUD} Lemmi per Emozione: {label} (Log Odds Ratio)')
    plt.axis('off')
    file_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    plt.savefig(os.path.join(output_dir_wordclouds, f'wordcloud_{file_label}.png'))
    print(f"Word cloud per '{label}' salvata come '{output_dir_wordclouds}/wordcloud_{file_label}.png'.")
    plt.close()

# --- Funzione per Creare la GIF dalle Word Clouds ---
def create_gif_from_wordclouds(image_folder, output_gif, duration):
    """
    Crea una GIF animata dalle immagini PNG in una cartella specificata.
    """
    image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))

    if not image_files:
        print(f"Nessuna immagine PNG trovata in '{image_folder}'. Impossibile creare la GIF.")
        return

    images = []
    for filename in image_files:
        try:
            images.append(imageio.imread(filename))
        except Exception as e:
            print(f"Errore nella lettura dell'immagine '{filename}': {e}")
            continue

    if not images:
        print("Nessuna immagine valida è stata caricata per la creazione della GIF.")
        return

    print(f"\nCreazione GIF: '{output_gif}' da {len(images)} immagini...")
    imageio.mimsave(output_gif, images, duration=duration, loop=0)
    print(f"GIF salvata con successo come '{output_gif}'")

# --- Chiama la funzione per creare la GIF ---
create_gif_from_wordclouds(output_dir_wordclouds, GIF_OUTPUT_FILENAME, GIF_FRAME_DURATION)

print("\nProcesso completato. Controlla le cartelle di output e il file GIF per i risultati.")