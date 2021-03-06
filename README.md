# Algeo02-20065
Website for image compression using SVD (Singular Value Decomposition) algorithm.

## Table of Contents
* [General Information](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Structures](#structures)
* [Setup](#setup)
* [Usage](#usage)

## General Information
Image compression is a type of data compression that applied to digital image. Using image compression, we can reduce the file size of an image to significant levels without decreasing the quality of the image itself. There are numerous algorithm that are used for image compression, for example SVD (Singular Value Decomposition), DCT (Discrete Cosine Transform), and LZW (Lempel–Ziv–Welch). In this case, we used SVD for the algorithm in this program.
SVD is a method for factoring matrix into three submatrix (orthogonal matrix U, diagonal matrix Σ, and transpose of orthogonal matrix V). The method in SVD itself is a generalization from eigendecomposition of a square matrix.

## Technologies Used
* Flask - version 2.0.2
* opencv - version 4.5.48
* Pillow - 8.4.0

## Features
* Image compression with user input compression percentage
* Compressed image preview
* Lossless image saving

## Screenshots
### Tampilan Front Page
![Front Page](./test/website/1.png)
### Tampilan View Page
![View Page](./test/website/2.png)

## Structures
```bash
.
│   README.md
│   requirements.txt
│
├───src
│   │   app.py
│   │   Eigen.py
│   │   SVD.py
│   │
│   ├───static
│   │   ├───images
│   │   │       imageNotAvailable.png
│   │   │       logo.png
│   │   │       logoSmall.png
│   │   │
│   │   └───styles
│   │           frontpage.css
│   │
│   └───templates
│           frontpage.html
│
└───test
    ├───pdf
    │   ├───1
    │   │       100%_ori.jpg
    │   │       2%.jpg
    │   │       20%.jpg
    │   │       40%.jpg
    │   │       80%.jpg
    │   │
    │   ├───2
    │   │       100%_ori.jpg
    │   │       2%.jpg
    │   │       20%.jpg
    │   │       40%.jpg
    │   │       80%.jpg
    │   │
    │   ├───3
    │   │       100%_ori.jpg
    │   │       2%.jpg
    │   │       20%.jpg
    │   │       40%.jpg
    │   │       80%.jpg
    │   │
    │   ├───4
    │   │       100%_ori.jpg
    │   │       2%.jpg
    │   │       20%.jpg
    │   │       40%.jpg
    │   │       80%.jpg
    │   │
    │   ├───5
    │   │       100%_ori.png
    │   │       2%.png
    │   │       20%.png
    │   │       40%.png
    │   │       80%.png
    │   │
    │   └───6
    │           100%_ori.png
    │           2%.png
    │           20%.png
    │           40%.png
    │           80%.png
    │
    └───website
            1.png
            2.png
```

## Setup
1. Go to src folder.
2. Create new virtual environment in that folder using `virtualenv venv`.
3. Activate aforementioned environment using `venv/bin/activate` on Linux/OS X and `venv\scripts\activate` on Windows.
4. Install all dependencies using `pip install -r requirements.txt`.

## Usage
1. Run `App.py` from text editor.
2. Open `127.0.0.1:5000` localhost from your browser.
3. Upload and compress your image from the webpage.
4. Close terminal to quit from program.
