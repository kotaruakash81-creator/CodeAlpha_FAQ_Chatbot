"""
FAQ Chatbot
-----------
- Loads a set of FAQs (question, answer) from a CSV file.
- Preprocesses text (tokenize, lowercase, remove stopwords/punctuation, lemmatize).
- Uses TF-IDF + cosine similarity to match a user's question to the closest FAQ.
- Returns the best-matching answer, or a fallback message if nothing matches well.

Uses NLTK for preprocessing when it's available (with its data downloaded).
If NLTK or its data isn't available, it automatically falls back to a
lightweight built-in tokenizer/stopword-remover, so the chatbot still runs.
"""

import re
import csv
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# 1. Preprocessing
# --------------------------------------------------------------------------

# A small built-in stopword list used as a fallback if NLTK data isn't present.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "do", "does", "did", "doing", "have", "has", "had",
    "having", "in", "on", "at", "to", "for", "of", "with", "about", "as",
    "by", "and", "or", "but", "if", "so", "than", "too", "very", "can",
    "could", "will", "would", "should", "what", "which", "who", "whom",
    "how", "when", "where", "why", "there", "here", "not", "no", "just",
}

_USE_NLTK = False
_lemmatizer = None
_stopwords = _FALLBACK_STOPWORDS

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Make sure the required NLTK data packages are present; download quietly
    # if missing (requires internet on first run only).
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(
                f"tokenizers/{pkg}" if "punkt" in pkg else
                f"corpora/{pkg}"
            )
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    # Verify everything actually works end-to-end before committing to NLTK.
    word_tokenize("test sentence")
    _stopwords = set(nltk_stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _USE_NLTK = True
except Exception:
    _USE_NLTK = False


def preprocess(text: str) -> str:
    """Clean, tokenize, remove stopwords/punctuation, and lemmatize/normalize text."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # strip punctuation/symbols

    if _USE_NLTK:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
        tokens = [
            _lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in _stopwords and tok not in string.punctuation and tok.strip()
        ]
    else:
        tokens = [
            tok for tok in text.split()
            if tok not in _stopwords and tok.strip()
        ]

    return " ".join(tokens)


# --------------------------------------------------------------------------
# 2. FAQ Chatbot class
# --------------------------------------------------------------------------

class FAQChatbot:
    def __init__(self, csv_path: str, similarity_threshold: float = 0.25):
        self.questions = []
        self.answers = []
        self.similarity_threshold = similarity_threshold

        self._load_faqs(csv_path)

        # Preprocess all FAQ questions once, up front.
        self._processed_questions = [preprocess(q) for q in self.questions]

        # Build the TF-IDF matrix for the FAQ questions.
        self.vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self.vectorizer.fit_transform(self._processed_questions)

    def _load_faqs(self, csv_path: str):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.questions.append(row["question"].strip())
                self.answers.append(row["answer"].strip())

    def get_response(self, user_query: str):
        """Return (answer, matched_question, similarity_score) for the closest FAQ."""
        if not user_query.strip():
            return "Please type a question so I can help.", None, 0.0

        processed_query = preprocess(user_query)
        query_vec = self.vectorizer.transform([processed_query])

        similarities = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        if best_score < self.similarity_threshold:
            return (
                "Sorry, I couldn't find a good match for that question. "
                "Try rephrasing it, or ask something else about Python.",
                None,
                float(best_score),
            )

        return self.answers[best_idx], self.questions[best_idx], float(best_score)


# --------------------------------------------------------------------------
# 3. Simple command-line chat loop
# --------------------------------------------------------------------------

def main():
    bot = FAQChatbot("faqs.csv")
    print("FAQ Chatbot (Python topics) — type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Bot: Goodbye!")
            break

        answer, matched_q, score = bot.get_response(user_input)
        if matched_q:
            print(f"Bot: {answer}")
            print(f"     (matched: \"{matched_q}\" | similarity: {score:.2f})\n")
        else:
            print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
