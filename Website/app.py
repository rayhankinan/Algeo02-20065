from flask import Flask, render_template, request, send_file
from werkzeug.utils import redirect, secure_filename
from SVD import compress
import numpy as np
import io
import cv2
import os
import base64
import time
import sys # BUAT CONSOLE LOG CERITANYA

app = Flask(__name__)
imageFile = None
imageFileCompressed = None
photoFileName = None
photoFileExtension = None
compressionRate = None
executionTime = None

@app.route('/', methods=['GET'])
def front_page():
    return render_template('frontpage.html', byteImage = None, byteImageCompressed = None, pixelDifference = None, compressionTime = None, fileName = None)

@app.route('/upload', methods=['POST'])
def upload_page():

    global imageFile
    global imageFileCompressed
    global photoFileName
    global photoFileExtension
    global compressionRate
    global executionTime

    try:
        photo = request.files['uploaded-image']
        photoFileName, photoFileExtension = os.path.splitext(secure_filename(photo.filename))

        inMemoryFile = io.BytesIO()
        photo.save(inMemoryFile)
        data = np.frombuffer(inMemoryFile.getvalue(), dtype=np.uint8)
        imageFile = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except:
        pass

    imageFileCompressed = None

    try:
        compressionRate = int(request.form['compression-rate'])
    except:
        compressionRate = 100

    return redirect('/compress')

@app.route('/view', methods=['GET'])
def view_page():

    global imageFile
    global imageFileCompressed
    global photoFileName
    global photoFileExtension
    global compressionRate
    global executionTime
    
    _, frameImage = cv2.imencode(photoFileExtension, imageFile)
    _, frameImageCompressed = cv2.imencode(photoFileExtension, imageFileCompressed)

    delta = cv2.absdiff(imageFile.astype(np.float32), imageFileCompressed.astype(np.float32))
    percentage = round((np.count_nonzero(delta) * 100) / delta.size, 2)

    return render_template('frontpage.html', byteImage = "data:image/" + photoFileExtension[1:] + ";base64," + base64.b64encode(frameImage).decode('utf-8'), byteImageCompressed = "data:image/" + photoFileExtension[1:] + ";base64," + base64.b64encode(frameImageCompressed).decode('utf-8'), pixelDifference = percentage, compressionTime = executionTime, fileName = photoFileName + photoFileExtension, persenKompresi = compressionRate)

@app.route('/save', methods=['POST'])
def save_image():

    global imageFile
    global imageFileCompressed
    global photoFileName
    global photoFileExtension
    global compressionRate
    global executionTime

    _, frameImageCompressed = cv2.imencode(photoFileExtension, imageFileCompressed)

    return send_file(io.BytesIO(frameImageCompressed), download_name=photoFileName + photoFileExtension, mimetype="images/" + photoFileExtension[1:])

@app.route('/remove', methods=['POST'])
def remove_image():

    global imageFile
    global imageFileCompressed
    global photoFileName
    global photoFileExtension
    global compressionRate
    global executionTime

    imageFile = None
    imageFileCompressed = None
    photoFileName = None
    photoFileExtension = None
    compressionRate = None

    return redirect('/')

@app.route('/compress', methods=['GET'])
def compress_image():

    global imageFile
    global imageFileCompressed
    global photoFileName
    global photoFileExtension
    global compressionRate
    global executionTime

    startTime = time.time()
    imageFileCompressed = compress(imageFile, compressionRate)
    executionTime = round(time.time() - startTime, 2)

    return redirect('/view')

if __name__ == '__main__':
    app.run(debug=True)