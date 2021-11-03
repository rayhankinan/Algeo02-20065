from flask import Flask, render_template, request
from werkzeug.utils import redirect, secure_filename
import os
import sys

UPLOAD_FOLDER = 'ImageFolder'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def front_page():
    return render_template('frontpage.html')

@app.route('/upload', methods=['POST'])
def upload_page():
    img = request.files['uploaded-image']
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(img.filename)))
    return redirect('/')

if __name__ == '__main__':
    print(os.getcwd(), file=sys.stdout)
    app.run(debug=True)