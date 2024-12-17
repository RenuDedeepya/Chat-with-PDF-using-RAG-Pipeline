# Chat-with-PDF-using-RAG-Pipeline

This project enables conversational interaction with PDF documents using the Retrieval-Augmented Generation (RAG) model. Users can ask questions related to the content of uploaded PDF files, and the system will provide detailed responses using RAG-based conversation generation techniques.

Features

PDF Text Extraction:

Ability to extract text from PDF documents using PyMuPDF (fitz).

Text Chunking:

Splits the extracted text into manageable chunks for processing, allowing for efficient embedding generation.

Sentence Embedding Generation:

Utilizes a pre-trained model from the Sentence Transformers library to convert text chunks into numerical embeddings that capture semantic meaning.

Efficient Similarity Search:

Implements FAISS (Facebook AI Similarity Search) to store and retrieve embeddings efficiently, enabling quick access to similar text chunks based on user queries.

Query Processing:

Allows users to input queries and retrieves the most relevant text chunks based on semantic similarity to the query.

Term Comparison:

Compares retrieved chunks against specific terms to identify relevant information, providing a way to filter results based on user-defined keywords.

Response Generation:

Generates a user-friendly response summarizing the relevant details based on the retrieved chunks and the original query.

Tech Stack

Programming Language:

Python: The primary language used for developing the application.

Libraries and Frameworks:

PyMuPDF (fitz): For handling PDF files and extracting text.

Sentence Transformers: For generating sentence embeddings from text.

FAISS: For efficient similarity search and clustering of embeddings.

NumPy: For numerical operations and handling arrays.

To create a requirements.txt file, you can simply create a new text file named requirements.txt and copy the above lines into it. You can then install the required packages using pip with the following command:

pip install -r requirements.txt
