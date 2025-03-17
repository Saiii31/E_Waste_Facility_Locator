from flask import Flask, render_template, request
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0:'Camera',1:'Keyboard',2:'Laptop',3:'Microwave',4:'Mobile',5:'Mouse',6:'Smartwatch',7:'TV'}

# model = keras.models.load_model('my_model.h5')
model = load_model('my_model.h5')
model.make_predict_function()

def predict_image(img_path):
  i = image.load_img(img_path, target_size=(100,100))
  i = image.img_to_array(i)/255.0
  i = i.reshape(1,100,100,3)
  # p = (model.predict(i) > 0.5).astype("int32")
  p = np.argmax(model.predict(i),axis=-1)
  return dic[p[0]]

#routes
@app.route("/", methods=['GET','POST'])
def main():
  return render_template("index.html")

@app.route("/about")
def about_page():
  return "Please subscribe!"

@app.route("/submit",methods=['GET','POST'])
def get_output():
  if request.method == 'POST':
    img = request.files['my_image']

    img_path = "static/" + img.filename
    img.save(img_path)
    p = predict_image(img_path)
  return render_template("index.html",prediction=p, img_path=img_path)
    
@app.route("/camera")
def camera():
  return render_template("camera.html")

@app.route("/keyboard")
def keyboard():
  return render_template("keyboard.html")

@app.route("/laptop")
def laptop():
  return render_template("laptop.html")

@app.route("/microwave")
def microwave():
  return render_template("microwave.html")

@app.route("/mobile")
def mobile():
  return render_template("mobile.html")

@app.route("/mouse")
def mouse():
  return render_template("mouse.html")

@app.route("/smartwatch")
def smartwatch():
  return render_template("smartwatch.html")

@app.route("/tv")
def tv():
  return render_template("tv.html")

if __name__ == '__main__':
  app.run(debug = False,port=81)
