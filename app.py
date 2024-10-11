from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import load_img
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.applications.vgg16 import decode_predictions
import shutil
import os
import sqlite3
import base64
# from  keras._tf_keras.keras.optimizers import Adam

# Load the model and weights
model = load_model('./model/proj_isd.h5')

# Initialize FastAPI app
app = FastAPI()


def get_db_connection():
    conn = sqlite3.connect('Vegetable.db')
    conn.row_factory = sqlite3.Row  # ทำให้สามารถเข้าถึงค่าผ่านชื่อคอลัมน์ได้
    return conn

app.mount("/static", StaticFiles(directory='static'), name="static")

# Define templates directory
templates = Jinja2Templates(directory="templates")
 
# Define the image categories
category = {
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: "Broccoli", 5: 'Cabbage', 6: 'Capsicum',
    7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: "Radish", 14: "Tomato"
}


@app.get("/", response_class=HTMLResponse)
async def start(request: Request):
    return templates.TemplateResponse("start.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, imagefile: UploadFile = File(...)):
    # Save the uploaded image file
    image_path = os.path.join('static', 'image', 'input_users', imagefile.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(imagefile.file, buffer)

    # Load and preprocess the image
    img_ = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    # Make the prediction
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    classification = f"{category[index]}"
    percent = "{:2.0f}%".format(prediction[0][index] * 100)

    conn = get_db_connection()
    vegetable = conn.execute('SELECT * FROM vegetables WHERE name = ?', (classification,)).fetchone()
    conn.close()
  

    if vegetable is None:
        raise HTTPException(status_code=404, detail="Vegetable not found")
    

    # จัดการ benefit
    def format_benefits(benefits):
        # แยกตามจุด (.)
        sentences = benefits.split('.')
        # ลบช่องว่างหน้าหลังประโยค แล้วเติมขีดหน้าแต่ละบรรทัด
        return ['- ' + sentence.strip() for sentence in sentences if sentence.strip()]
    
    if vegetable:
        formatted_benefits = format_benefits(vegetable['benefits'])
        formatted_howtochoose = format_benefits(vegetable['how_to_choose'])
    else:
        formatted_benefits = None


    # จัดการ minerals
    minerals_descriptions = {
    "Calcium": "Strengthens bones and teeth.",
    "Iron": "Supports red blood cell production and prevents anemia.",
    "Potassium": "Helps regulate blood pressure and muscle function.",
    "Magnesium": "Assists in energy metabolism and maintaining blood sugar levels."
    }

    def get_mineral_details(minerals):
        # แยกสารอาหารออกจาก database (กรณีแยกด้วย ',')
        minerals_list = [mineral.strip() for mineral in minerals.split(',')]
        
        # สร้างรายการของคำอธิบายสำหรับแต่ละสารอาหาร
        details = []
        for minerals in minerals_list:
            if minerals in minerals_descriptions:
                details.append(f"- {minerals}: {minerals_descriptions[minerals]}")
        
        # แยกบรรทัดด้วยการเว้นบรรทัด
        return '<br>'.join(details)

    if vegetable:
        minerals_details = get_mineral_details(vegetable['minerals'])
        storage = get_mineral_details(vegetable['storage'])
    else:
        minerals_details = "No nutrition information available."

    #จัดการ blob ส่วนของรูปภาพ
    def convert_blob_to_base64(blob_data):
        return base64.b64encode(blob_data).decode('utf-8')

    # แปลง Row เป็น dict ก่อน
    vegetable = dict(vegetable) 
   
        # แปลงรูปภาพจาก BLOB เป็น base64 สำหรับแต่ละรูป
    for i in range(1, 7):
        pic_column = f"menu{i}_pic"
        if vegetable[pic_column] is not None:
            vegetable[pic_column] = convert_blob_to_base64(vegetable[pic_column])
    
    vegetable['image'] = convert_blob_to_base64(vegetable['image'])


    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": classification,
        "img_path": image_path,
        "vegetable": vegetable,
        "formatted_benefits": formatted_benefits,
        "minerals_details": minerals_details,
        "formatted_howtochoose" : formatted_howtochoose,
        "storage" : storage,
        "percent" : percent
    })