# Free PDF RAG lab for Google Colab
# Uses a local open-source model; no paid API key required.
# Note: RAG does not train/fine-tune the model. It indexes your PDF for retrieval.

!pip -q install pypdf scikit-learn transformers sentencepiece

from google.colab import files
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re

# 1) Upload a PDF
uploaded = files.upload()
pdf_path = next(name for name in uploaded if name.lower().endswith(".pdf"))

# 2) Read PDF text
reader = PdfReader(pdf_path)
pages = [page.extract_text() or "" for page in reader.pages]
pdf_text = "\n".join(pages)
if not pdf_text.strip():
    raise ValueError("No selectable text found. This may be a scanned PDF; run OCR first.")

# 3) Split text into overlapping chunks
def make_chunks(text, chunk_words=180, overlap_words=35):
    words = text.split()
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

chunks = make_chunks(pdf_text)
print(f"Indexed {len(chunks)} chunks from {len(reader.pages)} PDF pages.")

# 4) Build a free local retrieval index (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
chunk_matrix = vectorizer.fit_transform(chunks)

# 5) Load a small, free, open-source text generation model
# First run downloads model files; Colab may need a few minutes.
generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

# 6) Ask questions against your PDF
def ask_rag(question, top_k=3):
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, chunk_matrix).ravel()
    best_ids = similarities.argsort()[-min(top_k, len(chunks)):][::-1]
    context = "\n\n".join(chunks[i] for i in best_ids)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you could not find it in the PDF.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = generator(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"]
    return answer, [(int(i), float(similarities[i]), chunks[i]) for i in best_ids]

question = input("Ask a question about your PDF: ")
answer, sources = ask_rag(question)
print("\nANSWER:\n", answer)
print("\nRETRIEVED SOURCES:")
for rank, (chunk_id, score, chunk) in enumerate(sources, start=1):
    print(f"\n[{rank}] chunk={chunk_id}, relevance={score:.3f}\n{chunk[:900]}")

# To ask another question, run:
# answer, sources = ask_rag("Write your next question here")
# print(answer)
