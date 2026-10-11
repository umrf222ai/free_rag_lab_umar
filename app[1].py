import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch

st.set_page_config(page_title="Free PDF RAG Lab", page_icon="📚", layout="wide")
st.title("📚 Free PDF RAG Lab")
st.caption("Ask questions about a PDF using local open-source AI. No paid API key is needed.")

@st.cache_resource(show_spinner="Loading the free language model…")
def load_generator():
    # CPU inference is compatible with basic hosted environments, but may be slow.
    return pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page_texts).strip()
    if not text:
        raise ValueError("No selectable text was found. This may be a scanned PDF; OCR it first.")
    return text, len(reader.pages)

def make_chunks(text, chunk_words=180, overlap_words=35):
    words = text.split()
    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        part = " ".join(words[start:start + chunk_words])
        if part.strip():
            chunks.append(part)
    return chunks

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf:
    try:
        text, page_count = extract_pdf(uploaded_pdf)
        chunks = make_chunks(text)
        if not chunks:
            st.error("The PDF did not contain usable text.")
            st.stop()

        vectorizer = TfidfVectorizer(stop_words="english")
        chunk_matrix = vectorizer.fit_transform(chunks)

        st.success(f"PDF read successfully: {page_count} pages, {len(chunks)} text chunks indexed.")
        st.info("RAG does not retrain the AI model. It retrieves relevant passages from this PDF for each question.")

        question = st.text_input("Ask a question about your PDF")
        if st.button("Get answer", type="primary", disabled=not question.strip()):
            with st.spinner("Searching the PDF and generating an answer…"):
                q_vector = vectorizer.transform([question])
                scores = cosine_similarity(q_vector, chunk_matrix).ravel()
                top_ids = scores.argsort()[-min(3, len(chunks)):][::-1]
                context = "\n\n".join(chunks[i] for i in top_ids)
                prompt = (
                    "Answer the question using only the context below. "
                    "If the answer is not in the context, say you could not find it in the PDF.\n\n"
                    f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                )
                generator = load_generator()
                answer = generator(prompt, max_new_tokens=160, do_sample=False)[0]["generated_text"]
            st.subheader("Answer")
            st.write(answer)
            with st.expander("Show retrieved source passages"):
                for rank, idx in enumerate(top_ids, start=1):
                    st.markdown(f"**Source {rank} — chunk {idx + 1} (relevance {scores[idx]:.3f})**")
                    st.write(chunks[idx])
    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")
else:
    st.markdown("### How it works")
    st.markdown("1. Upload a PDF.  2. The app extracts and chunks its text.  3. TF-IDF retrieves relevant passages.  4. FLAN-T5 generates an answer from those passages.")
    st.warning("Text-based PDFs are supported. Scanned/image-only PDFs need OCR before use.")
