from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from flask import Flask, render_template,request
import numpy as np
app = Flask(__name__,static_url_path="")

model = load_model('bestmodel.h5')


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.form['image']
        img = load_img(f)
        img = img.resize((256,256))
        img = img_to_array(img)
        img = img.reshape((1,256,256,3))
        img = img/255
        pred = model.predict([img])
        class_label = np.argmax(pred)
        if class_label == 0:
            pred = 'Bacterial Leaf Blight'
        if class_label == 1:
            pred = 'Brown Spot'
        else:
            pred = 'Leaf Smut'
        return render_template('predict.html',predict=pred)
app.run(debug=True)