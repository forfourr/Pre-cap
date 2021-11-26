from flask import Flask, render_template, request, session, redirect, url_for, jsonify, json
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import ShowColor
import argparse
import os
from PIL import Image
app = Flask(__name__)
from Personal_color import analysis

@app.route("/")
def hello():
    return render_template('main.html')

############# main ######################
@app.route("/main/")
def main() :
    return render_template('main.html')


## upload
@app.route("/upload", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f'static/uploads/{secure_filename(f.filename)}')
        img = Image.open(f)
        tone = Personal_color.analysis(img)

        return render_template('main.html',tone = tone)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)