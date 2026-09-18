# FAQ Chatbot (Python Topics)

A simple FAQ chatbot that matches a user's question to the closest FAQ using
NLP preprocessing + TF-IDF cosine similarity, with a Flask-based chat UI.

## How it works
1. **`faqs.csv`** — 20 sample Python-programming FAQs (question, answer). Swap
   this file out with your own topic/product FAQs — just keep the same
   `question,answer` column format.
2. **`chatbot.py`** — Core logic:
   - `preprocess()` cleans and tokenizes text (lowercasing, removing
     punctuation/stopwords, lemmatizing). It uses **NLTK** if available (and
     downloads the small data packages it needs on first run); otherwise it
     automatically falls back to a lightweight built-in tokenizer, so the
     bot still works with zero setup.
   - `FAQChatbot` builds a **TF-IDF** matrix of all FAQ questions
     (scikit-learn) and matches a new user question against it using
     **cosine similarity**. If the best match is below a similarity
     threshold (default `0.25`), it returns a "couldn't find a match"
     fallback instead of a wrong answer.
   - Running `chatbot.py` directly gives you a command-line chat loop.
3. **`app.py`** + **`templates/index.html`** — Optional simple web chat UI
   (Flask backend, HTML/CSS/JS frontend) that calls the same matching logic
   through a `/ask` JSON endpoint.

## Setup

```bash
pip install -r requirements.txt
```

The first time you run it, NLTK will try to download a few small data
packages (`punkt`, `stopwords`, `wordnet`) — this needs an internet
connection once. If it can't download them (e.g. no internet), the chatbot
automatically uses its built-in fallback preprocessing instead, so it still
works.

## Run it

**Command line:**
```bash
python chatbot.py
```

**Web UI:**
```bash
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

## Customizing
- Replace the rows in `faqs.csv` with your own topic/product's FAQs.
- Adjust `similarity_threshold` in `FAQChatbot.__init__` (in `chatbot.py`) to
  make matching stricter (higher, fewer false matches) or looser (lower,
  more matches but possibly less accurate).
- Add more preprocessing steps (e.g. spelling correction, synonym expansion)
  in `preprocess()` if you want more advanced matching.
