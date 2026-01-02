from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained models
eye_model = load_model("models/eye_disease_model (1).h5")
tongue_model = load_model("models/tongue_disease_model.h5")

# Preprocessing function
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form(...)):
    try:
        contents = await file.read()
        image_array = preprocess_image(contents)

        if model_type == "eye":
            prediction = eye_model.predict(image_array)
        elif model_type == "tongue":
            prediction = tongue_model.predict(image_array)
        else:
            return {"error": "Invalid model_type. Use 'eye' or 'tongue'"}

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))


        print("predicted class:", predicted_class, "confidence:", confidence)
        
        if model_type == "eye":
            #labels = ["Anemia", "Jaundice", "Normal"]
            labels = ["Anemia", "Jaundice", "Normal"]
        else:
            #labels = ['Diabetes', 'Eldery Annotated', 'Atrophic Glossitics', 'Normal', 'Tounge DX']
            labels = [ 'Normal', 'B12 Deficiency', 'Diabetes','Altrophic Glossitics']

        #print(f"Predicted class: {labels[predicted_class]}, Confidence: {confidence}")
        #print(type(labels[predicted_class]), type(confidence))
        
        return {"class": labels[predicted_class], "confidence": round(confidence-0.2, 4)}

    except Exception as e:
        return {"error": str(e)}



# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import io

# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load models
# eye_model = load_model("models/mm_eye_model.h5")
# tongue_model = load_model("models/mm_tongue_model.h5")

# # Label mappings
# eye_labels = ["Anemia", "Jaundice", "Normal"]
# tongue_labels = ['Diabetes', 'Eldery Annotated','Atrophic Glossitics', 'Normal', 'Tounge DX']

# def preprocess_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize((150, 150))
#     img_array = np.array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# @app.post("/predict")
# async def predict(file: UploadFile = File(...), model_type: str = Form(...)):
#     try:
#         contents = await file.read()
#         image_array = preprocess_image(contents)

#         if model_type == "eye":
#             prediction = eye_model.predict(image_array)
#             labels = eye_labels
#         elif model_type == "tongue":
#             prediction = tongue_model.predict(image_array)
#             labels = tongue_labels
#         else:
#             return {"error": "Invalid model_type. Use 'eye' or 'tongue'"}

#         predicted_index = int(np.argmax(prediction))
#         predicted_class = labels[predicted_index]
#         confidence = round(float(np.max(prediction)), 4)
            

#         print(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        
#         return {"class": predicted_class, "confidence": confidence}

#     except Exception as e:
#         return {"error": str(e)}







# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# from io import BytesIO

# app = FastAPI()

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load models once
# eye_model = load_model("models\mm_eye_model.h5")
# tongue_model = load_model("models\mm_tongue_model.h5")

# eye_labels = ['Anemia', 'Jaundice', 'Normal']
# tongue_labels = ['Diabetes', 'Eldery Annotated', 'Intellisense', 'Normal', 'Tounge DX']

# def read_imagefile(file) -> Image.Image:
#     image_data = file.read()
#     return Image.open(BytesIO(image_data)).convert("RGB")

# def prepare_image(img: Image.Image) -> np.ndarray:
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#     return img_array

# @app.post("/predict")
# async def predict(file: UploadFile = File(...), model_type: str = Form(...)):
#     img = read_imagefile(await file.read())
#     img_tensor = prepare_image(img)

#     if model_type == "eye":
#         model = eye_model
#         labels = eye_labels
#     elif model_type == "tongue":
#         model = tongue_model
#         labels = tongue_labels
#     else:
#         return {"error": "Invalid model_type"}

#     predictions = model.predict(img_tensor)
#     confidence = float(np.max(predictions))
#     predicted_label = labels[np.argmax(predictions)]



#     print(f"Predicted class: {predicted_label}, Confidence: {confidence}")


#     return {
#         "predicted_label": predicted_label,
#         "confidence": confidence
#     }
