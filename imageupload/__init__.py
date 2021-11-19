from flask import Flask, render_template, request, session, redirect, url_for, jsonify, json
from werkzeug.utils import secure_filename


app = Flask(__name__)


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
        return render_template('main_style.html')
        img_pil = Image.open(f).convert("RGB")  ## PIL Image
        img_cv = np.array(img_pil)
        img_cv = img_cv[:, :, ::-1].copy()  ## RGB  ## CV Image

        face_img = face_detection(img_cv)
        # img_cv_bgr = cv.cvtColor(img_cv, cv.COLOR_RGB2BGR)
        # face_img = face_detection(img_cv_bgr)
        if np.array_equal(face_img, [0, 0, 0]):
            print(face_img)
            print("에러")
            return jsonify({"pctype": 'error'})
        else:
            success, img_cv_binary = cv.imencode('.jpg', img_cv)  ## Binary Image
            # plt.imshow(img_cv)
            # plt.show()
            print(success)
            if success:
                # img_db_save(img_cv_binary.tostring())
                ext_img = skin_extract(face_img)
                hsv, rgb = color_convert(ext_img)
                result = color_classifier(hsv)  # <<<<<<<<<<<---------------------------algorithm 수정부위

        print(result)
        print(color_type[result])
        result = {"pctype": color_type[result], "rgb_value": rgb}
        return jsonify(result)





if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)