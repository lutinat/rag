from flask import Flask, send_file, request, jsonify
from flask_cors import CORS
import rag

app = Flask(__name__)
CORS(app)

@app.route('/api/question', methods=['POST'])
def question():
    data = request.json
    question = data.get('question')
    answer, sources = rag.rag(question)
    print('Answer : ', answer)
    print('Sources : ', sources)
    return jsonify({
        'answer': answer,
        'sources': sources
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
