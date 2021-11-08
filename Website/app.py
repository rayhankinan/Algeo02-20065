import cv2
from flask import Flask, render_template, request
from werkzeug.utils import redirect
import numpy as np
import io
import cv2
import Eigen
import sys

from Eigen import eigenValue # BUAT CONSOLE LOG CERITANYA

app = Flask(__name__)
app.config['IMAGE_FILE'] = None

@app.route('/', methods=['GET'])
def front_page():
    return render_template('frontpage.html')

@app.route('/upload', methods=['POST'])
def upload_page():

    photo = request.files['uploaded-image']

    inMemoryFile = io.BytesIO()
    photo.save(inMemoryFile)
    data = np.fromstring(inMemoryFile.getvalue(), dtype=np.uint8)
    imageFile = cv2.imdecode(data, 1)
    
    #cv2.imshow("img_decode", imageFile)
    #cv2.waitKey()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)

