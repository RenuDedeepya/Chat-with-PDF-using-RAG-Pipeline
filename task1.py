import fitz  # for PDF handling
from sentence_transformers import SentenceTransformer #for text embeddings
import faiss   #for efficient similarity search
import numpy as np    #for numerical operations

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() # Extract text from the page
    return text

# chunk the text into smaller pieces
def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Convert chunks to embeddings using the pre-trained model
def get_embeddings(chunks):
    return model.encode(chunks)

# Store embeddings in FAISS index for efficient retrieval
def store_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Path to the PDF file
pdf_path = "C:/Users/renud/Desktop/sitaphal/material1.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
embeddings = get_embeddings(chunks)
index = store_embeddings(np.array(embeddings))

#to query the embeddings and retrieve similar chunks
def query_embeddings(query, model, index):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=5)  # Retrieve top 5 similar chunks
    return I  

# querying the embeddings
query = "summarize types of visualization of data"
retrieved_indices = query_embeddings(query, model, index)
retrieved_chunks = [chunks[i] for i in retrieved_indices[0]]

#to compare retrieved chunks based on specific terms
def compare_chunks(chunks, terms):
    # This is a placeholder for actual comparison logic
    comparison_results = {}
    for term in terms:
        comparison_results[term] = [chunk for chunk in chunks if term in chunk]
    return comparison_results

#comparing chunks with specific terms
comparison_terms = ["Bachelor's Degree", "Master's Degree"]
comparison_results = compare_chunks(retrieved_chunks, comparison_terms)

#to generate a response based on retrieved chunks and the query
def generate_response(retrieved_chunks, query):
    # Placeholder for LLM response generation
    response = f"Based on your query '{query}', here are the relevant details:\n"
    for chunk in retrieved_chunks:
        response += f"- {chunk}\n"
    return response

response = generate_response(retrieved_chunks, query)
print(response)