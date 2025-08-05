from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)
generator = pipeline("text-generation", model="gpt2")

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.form.get("prompt", "")
    if not prompt:
        return render_template("index.html", response="Lütfen bir metin girin.")
    
    result = generator(prompt, max_length=100, num_return_sequences=1)
    generated = result[0]['generated_text']
    return render_template("index.html", response=generated)

if __name__ == '__main__':
    app.run(debug=True)
