import cv2
from flask import Flask, render_template, request
from flask.helpers import url_for
from werkzeug.utils import redirect, secure_filename
import numpy as np
import io
import cv2
import os
import base64
import sys # BUAT CONSOLE LOG CERITANYA

app = Flask(__name__)
imageFile = None
imageFileCompressed = None
photoFileName = None
photoFileExtension = None
compressionRate = None

@app.route('/', methods=['GET'])
def front_page():
    return render_template('frontpage.html', byteImage = None, byteImageCompressed = None, pixelDifference = None, compressionTime = None)

@app.route('/upload', methods=['POST'])
def upload_page():

    global imageFile
    global imageFileCompressed
    global photoFileName
    global photoFileExtension
    global compressionRate

    photo = request.files['uploaded-image']
    photoFileName, photoFileExtension = os.path.splitext(secure_filename(photo.filename))

    compressionRate = int(request.form['compression-rate']) / 100

    inMemoryFile = io.BytesIO()
    photo.save(inMemoryFile)
    data = np.frombuffer(inMemoryFile.getvalue(), dtype=np.uint8)
    imageFile = cv2.imdecode(data, 1)

    imageFileCompressed = None

    return redirect('/view')

@app.route('/view', methods=['GET'])
def view_page():

    global imageFile
    global imageFileCompressed
    global photoFileName
    global photoFileExtension
    
    if imageFileCompressed == None:
        _, frameImage = cv2.imencode(photoFileExtension, imageFile)
        return render_template('frontpage.html', byteImage = "data:image/" + photoFileExtension[1:] + ";base64," + base64.b64encode(frameImage).decode('utf-8'), byteImageCompressed = None, pixelDifference = None, compressionTime = None)
    else:
        _, frameImage = cv2.imencode(photoFileExtension, imageFile)
        _, frameImageCompressed = cv2.imencode(photoFileExtension, imageFileCompressed)

        delta = cv2.absdiff(imageFile, imageFileCompressed).astype(np.uint8)
        percentage = round((np.count_nonzero(delta) * 100) / delta, 2)

        waktuEksekusi = 0 # INI NANTI DIUBAH DENGAN WAKTU EKSEKUSI

        return render_template('frontpage.html', byteImage = "data:image/" + photoFileExtension[1:] + ";base64," + base64.b64encode(frameImage).decode('utf-8'), byteImageCompressed = "data:image/" + photoFileExtension[1:] + ";base64," + base64.b64encode(frameImageCompressed).decode('utf-8'), pixelDifference = percentage, compressionTime = waktuEksekusi)

@app.route('/save', methods=['POST'])
def save_image():
    pass

@app.route('/compress', methods=['POST'])
def compress_image():
    pass

if __name__ == '__main__':
    app.run(debug=True)

