# Algeo02-20065
Website for image compression using SVD (Singular Value Decomposition) algorithm.

## General Information
Image compression is a type of data compression that applied to digital image. Using image compression, we can reduce the file size of an image to significant levels without decreasing the quality of the image itself. There are numerous algorithm that are used for image compression, for example SVD (Singular Value Decomposition), DCT (Discrete Cosine Transform), and LZW (Lempel–Ziv–Welch). In this case, we used SVD for the algorithm that we used in this program. <br/>
SVD is a method for factoring matrix into three submatrix (orthogonal matrix U, diagonal matrix $\sigma$, and transpose of orthogonal matrix V). The method in SVD itself is a generalization from eigendecomposition of a square matrix. 

## Technologies Used
Flask - version 2.0.2 <br/>
opencv - version 4.5.48 <br/>
Pillow - 8.4.0 <br />

## Features
* Image compression with user input compression percentage
* Compressed image preview
* Lossless image saving

## Structures
|   output.txt
|   README.md
|   requirement.txt
|   
\---Website
    |   app.py
    |   Eigen.py
    |   SVD.py
    |   
    +---static
    |   +---images
    |   |       imageNotAvailable.png
    |   |       logo.png
    |   |       logoSmall.png
    |   |       
    |   \---styles
    |           frontpage.css
    |           
    \---templates
            frontpage.html