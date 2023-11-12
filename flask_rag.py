from flask import Flask, request, jsonify

from sentence_transformers import SentenceTransformer

from sklearn.neighbors import NearestNeighbors

from bert_score import score


from docx import Document

app = Flask(__name__)

def chunk_by_sentence(text, max_chunk_length=200):
    paragraphs = text.split(".")  # Assuming sentences are separated by full stop.
    chunks = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_length:
            chunks.append(paragraph)
        else:
            # For long sentences, further split them into smaller chunks
            words = paragraph.split()
            for i in range(0, len(words), max_chunk_length):
                chunk = " ".join(words[i:i+max_chunk_length])
                chunks.append(chunk)
                
    return chunks

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

vastu_text = extract_text_from_docx('vastu-shastra-processed.docx')
# We need to clean the document for un-necessary symbols so that the api calls can be optimised
def clean_text(text):
    cleaned_text = ''
    for char in text:
        if char.isalnum() or char.isspace() or char == '.':
            cleaned_text += char
    return cleaned_text

vastu_text = clean_text(vastu_text)

def query_rag(question, model, vector_store, document_chunks):
    question_embedding = model.encode([question])
    _, indices = vector_store.kneighbors(question_embedding)
    return document_chunks[indices[0][0]]

document_chunks = chunk_by_sentence(vastu_text)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(document_chunks)

vector_store = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
vector_store.fit(embeddings)


def compare_responses(baseline_response, rag_response):
    P, R, F1 = score([rag_response], [baseline_response], lang='en')
    return P.mean(), R.mean(), F1.mean()


@app.route('/rag', methods=['POST'])
def rag_response():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    question = data['question']
    answer_rag = query_rag(question, model, vector_store, document_chunks)

    return jsonify({'answer_rag': answer_rag})

@app.route('/bart_score', methods=['POST'])
def bart_score():
    data = request.json
    if not data or 'answer_openai' not in data or 'answer_rag' not in data:
        return jsonify({'error': 'No question provided'}), 400
    P, R, F1 = compare_responses(data['answer_openai'], data['answer_rag'])

    return jsonify({'Precision': P.numpy().tolist(), 'Recall': R.numpy().tolist(), 'F1 score':F1.numpy().tolist()})

if __name__ == '__main__':
    app.run(debug=True)
