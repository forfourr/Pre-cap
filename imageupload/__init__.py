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
        return render_template('main.html')


############# menu2
@app.route("/color_recommnd")
def color_recommnd():
    global rgb
    if len(rgb) == 3:
        rgb_value = rgb
        print(rgb_value)
        hsv = color_convert_hsv(rgb_value)
        print(hsv)
        hsv_palette_bright = CP.palette_bright(hsv)
        hsv_palette_harmony = CP.palette_harmony(hsv)

        rgb_palette_bright = CP.to_rgb(hsv_palette_bright)
        rgb_palette_harmony = CP.to_rgb(hsv_palette_harmony)

        data = {"bright": rgb_palette_bright, "harmony": rgb_palette_harmony}
        print(data['bright'][0])
        print(data['harmony'][0])
        return render_template('CAI_palette.html', data=json.dumps(data))
    else:
        print("rgb값 없음")
        return render_template('CAI_palette.html')




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)