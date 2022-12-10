from flask import Flask,request
from flask import jsonify
from tensorflow import keras
import numpy as np
import Labels
import modelProcess as pr
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
UPLOAD_FOLDER = 'upload/'
PATH = 'upload/requestImg.jpg'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = keras.models.load_model('kanji.h5')

# @app.route("/")
# def home():
#     return jsonify({'name':'愛'})

@app.route('/model',methods = ["POST","GET"])
def predict():

    if request.method == "POST":
        # print("request data", np.asarray(request.files))
        try:
            data = request.form['data']
            # filename = secure_filename(data.filename)
            # data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # path = UPLOAD_FOLDER+filename
            # img = pr.readAndProcessImg(path)
            # print(img)
            # print(type(data))
            pr.covertBase64ToImg(data)
            img = pr.readAndProcessImg(PATH)
            # print(np.asarray(img))
            result = model.predict(img.reshape(1,48,48,1))
            arr_predict = pr.sortPredict(result)

            pr.remove_file('./upload')
            return jsonify({'1':(arr_predict[-1][1]),
                            '2':(arr_predict[-2][1]),
                            '3':(arr_predict[-3][1]),
                            '4':(arr_predict[-4][1]),
                            '5':(arr_predict[-5][1]),
                            '6':(arr_predict[-6][1]),
                            '7':(arr_predict[-7][1]),
                            '8':(arr_predict[-8][1]),
                            '9':(arr_predict[-9][1])
            })
        except Exception as e:
            return jsonify(e)
    if request.method == "GET":
        try:
            return jsonify(Labels.labels)
        except Exception as e:
            return jsonify(e)
if __name__ == "__main__":
    app.run(debug=True)