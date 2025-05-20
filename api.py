from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
# import rag

app = Flask(__name__)
CORS(app)

@app.route('/api/question', methods=['POST'])
def question():
    data = request.json
    question = data.get('question')
    #answer, sources = rag.rag(question)
    answer = "This is a test answer"
    sources = [
        {"name": "Source 1", "url": "https://example.com/source1"},
        {"name": "Source 2", "url": "https://example.com/source2"}
    ]
    return jsonify({
        'answer': answer,
        'sources': sources
    })

if __name__ == '__main__':
    app.run(debug=True)
