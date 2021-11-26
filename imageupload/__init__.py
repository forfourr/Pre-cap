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

import cv2
import numpy as np
from colormath.color_objects import LabColor, sRGBColor, HSVColor
from colormath.color_conversions import convert_color

from ShowColor import DetectFace
from ShowColor import DominantColors
from ShowColor import tone_analysis
from ShowColor import white_balance


def analysis(imgpath):

    #######################################
    #           white balance             #
    #######################################
    img = cv2.imread(imgpath)
    wb = white_balance.balancing(img, 1)

    #######################################
    #           Face detection            #
    #######################################
    df = DetectFace("wb.jpg")
    face = [df.left_cheek, df.right_cheek,
            df.left_eyebrow, df.right_eyebrow,
            df.left_eye, df.right_eye]

    #######################################
    #         Get Dominant Colors         #
    #######################################
    temp = []
    clusters = 4
    for f in face:
        dc = DominantColors(f, clusters)
        face_part_color, _ = dc.getHistogram()
        #dc.plotHistogram()
        temp.append(np.array(face_part_color[0]))
    cheek = np.mean([temp[0], temp[1]], axis=0)
    eyebrow = np.mean([temp[2], temp[3]], axis=0)
    eye = np.mean([temp[4], temp[5]], axis=0)

    Lab_b, hsv_s = [], []
    color = [cheek, eyebrow, eye]
    for i in range(3):
        rgb = sRGBColor(color[i][0], color[i][1], color[i][2], is_upscaled=True)
        lab = convert_color(rgb, LabColor, through_rgb_type=sRGBColor)
        hsv = convert_color(rgb, HSVColor, through_rgb_type=sRGBColor)
        Lab_b.append(float(format(lab.lab_b,".2f")))
        hsv_s.append(float(format(hsv.hsv_s,".2f"))*100)

    print('Lab_b[skin, eyebrow, eye]',Lab_b)
    print('hsv_s[skin, eyebrow, eye]',hsv_s)
    #######################################
    #      Personal color Analysis        #
    #######################################
    Lab_weight = [30, 20, 5]
    hsv_weight = [10, 1, 1]
    if(tone_analysis.is_warm(Lab_b, Lab_weight)):
        if(tone_analysis.is_spr(hsv_s, hsv_weight)):
            tone = '봄웜톤(spring)'
        else:
            tone = '가을웜톤(fall)'
    else:
        if(tone_analysis.is_smr(hsv_s, hsv_weight)):
            tone = '여름쿨톤(summer)'
        else:
            tone = '겨울쿨톤(winter)'

    return tone

global tone
SW=ShowColor.white_balance()
SD=ShowColor.DetectFace()
SC=ShowColor.DominantColors()
SC=ShowColor.tone_analysis()



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
    global tone
    if request.method == 'POST':
        f = request.files['file']
        f.save(f'static/uploads/{secure_filename(f.filename)}')
        img = Image.open(f)

        tone = analysis(img)

        return render_template('main.html',tone = tone)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)