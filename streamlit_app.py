import streamlit as st
import spacy
import pytextrank
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect

@st.cache_resource
def load_spacy_model(lang):
    """
    Carica il modello spaCy appropriato per la lingua specificata e aggiunge pytextrank
    per l'estrazione delle parole chiave.
    """
    if lang == 'en':
        try:
            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe("textrank")
            return nlp
        except OSError:
            st.error("Il modello 'en_core_web_sm' non è stato trovato. Installalo eseguendo 'python -m spacy download en_core_web_sm'")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe("textrank")
            return nlp
    elif lang == 'it':
        try:
            nlp = spacy.load("it_core_news_sm")
            nlp.add_pipe("textrank")
            return nlp
        except OSError:
            st.error("Il modello 'it_core_news_sm' non è stato trovato. Installalo eseguendo 'python -m spacy download it_core_news_sm'")
            spacy.cli.download("it_core_news_sm")
            nlp = spacy.load("it_core_news_sm")
            nlp.add_pipe("textrank")
            return nlp
    else:
        st.error("Lingua non supportata. Usa 'en' per inglese o 'it' per italiano.")
        return None

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'

def get_language_name(lang_code):
    language_names = {
        'en': 'Inglese',
        'it': 'Italiano'
    }
    return language_names.get(lang_code, 'Lingua sconosciuta')

def extract_keywords(text, nlp):
    doc = nlp(text)
    keywords = [(phrase.text, phrase.rank) for phrase in doc._.phrases[:10]]
    return keywords

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

def display_sentiment_analysis(polarity, subjectivity):
    st.subheader("Sentiment Analysis")
    polarity_color = "green" if polarity > 0 else "red" if polarity < 0 else "gray"
    sentiment_html = f"""
    <style>
    .sentiment_table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1.5rem;
    }}
    .sentiment_table td, .sentiment_table th {{
        border: 1px solid #ddd;
        padding: 8px;
    }}
    </style>
    <table class="sentiment_table">
        <tr>
            <th>Aspetto</th>
            <th>Valore</th>
        </tr>
        <tr>
            <td>Polarità</td>
            <td><span style='color: {polarity_color};'>{polarity:.2f}</span></td>
        </tr>
        <tr>
            <td>Soggettività</td>
            <td><span>{subjectivity:.2f}</span></td>
        </tr>
    </table>
    """
    st.markdown(sentiment_html, unsafe_allow_html=True)

def plot_keyword_relevance(keywords):
    keywords_df = pd.DataFrame(keywords, columns=["Parola Chiave", "Punteggio"])
    keywords_df.sort_values(by="Punteggio", ascending=False, inplace=True)
    
    fig, ax = plt.subplots()
    ax.barh(keywords_df["Parola Chiave"], keywords_df["Punteggio"], color='skyblue')
    ax.set_xlabel('Punteggio')
    ax.set_ylabel('Parola Chiave')
    ax.set_title('Keyword Relevance Scores')
    ax.invert_yaxis()
    st.pyplot(fig)

def get_relevance_label(relevance):
    if relevance > 0.15:
        return 'Alta'
    elif relevance > 0.05:
        return 'Media'
    else:
        return 'Bassa'

def display_relevance_table(keywords):
    keywords_df = pd.DataFrame(keywords, columns=["Parola Chiave", "Punteggio"])
    keywords_df['Rilevanza'] = keywords_df['Punteggio'].apply(get_relevance_label)
    keywords_df.sort_values(by="Punteggio", ascending=False, inplace=True)
    
    st.subheader("Keyword Relevance Scores")
    st.dataframe(keywords_df)

    st.info("""
    **Cosa significano i punteggi?**
    
    - **Alta Rilevanza**: Parole chiave con punteggio > 0.15 sono considerate di alta rilevanza, indicando che sono molto significative nel contesto del documento.
    - **Media Rilevanza**: Parole chiave con punteggio tra 0.05 e 0.15 sono considerate di media rilevanza, indicando che sono abbastanza significative.
    - **Bassa Rilevanza**: Parole chiave con punteggio < 0.05 sono considerate di bassa rilevanza, indicando che hanno meno importanza nel contesto del documento.
    """)

col1, col2 = st.columns([1, 7])

with col2:
    st.title('Keyword Relevance & Sentiment Analyzer')
    st.markdown('###### by [NUR® Digital Marketing](https://www.nur.it)')

with st.expander("Introduzione"):
    st.markdown("""
    Questo strumento è progettato per offrire un'analisi dettagliata del testo, evidenziando le parole chiave più rilevanti e fornendo un'analisi del sentiment. L'obiettivo è supportare strategie di contenuto e SEO efficaci, consentendo una comprensione profonda sia dell'importanza di specifici termini (Keyword Relevance Score) che del tono emotivo generale del testo (Sentiment Analysis).
    """)

with st.expander("Istruzioni"):
    st.markdown("""
    Segui questi passaggi per analizzare il tuo testo:
    1. **Inserimento Testo**: Carica il testo che desideri analizzare.
    2. **Parole Chiave Target (Opzionale)**: Specifica eventuali parole chiave di interesse.
    3. **Analisi**: Clicca su "Analizza" per eseguire l'elaborazione.
    4. **Risultati**: Esamina i risultati per rilevanza delle parole chiave e sentiment del testo.
    """)

st.markdown("---")

user_input = st.text_area("Inserisci il testo qui:")
target_keywords_input = st.text_input(
    label="Parole Chiave Target",
    placeholder="Inserisci le Parole Chiave (separate da virgola)",
    label_visibility="collapsed"
)

if st.button("Analizza"):
    if user_input:
        lang = detect_language(user_input)
        st.info(f"Lingua rilevata: {get_language_name(lang)}")
        
        nlp = load_spacy_model(lang)
        keywords = extract_keywords(user_input, nlp)
        polarity, subjectivity = analyze_sentiment(user_input)
        
        target_keywords = [keyword.strip() for keyword in target_keywords_input.split(',') if keyword.strip()]
        target_keywords_relevance = [keyword for keyword in keywords if keyword[0] in target_keywords]
        
        st.subheader("Target Keyword Relevance")
        if target_keywords_relevance:
            target_keywords_df = pd.DataFrame(target_keywords_relevance, columns=["Parola Chiave Target", "Punteggio"])
            st.table(target_keywords_df)
        else:
            st.write("Nessuna delle parole chiave target trovata nel testo.")
        
        display_relevance_table(keywords)
        plot_keyword_relevance(keywords)
        display_sentiment_analysis(polarity, subjectivity)

        st.info("""
        **Cosa significano i punteggi?**
        
        - **Polarità**: Indica l'orientamento emotivo generale del testo, variando da -1 (molto negativo) a +1 (molto positivo). Un punteggio vicino a 0 indica un sentiment neutrale.
        
        - **Soggettività**: Misura quanto il testo esprime opinioni piuttosto che fatti, con un punteggio che va da 0 (molto oggettivo) a 1 (molto soggettivo).
        """)
