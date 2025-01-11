import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from fastapi.responses import StreamingResponse
import socket
from PIL import Image as PILImage

# Load the saved U-Net model
model = load_model("unet_wound_segmentation.h5")

# FastAPI app instance
app = FastAPI()

# Utility function for preprocessing images
def preprocess_image(image: PILImage.Image, target_size=(256, 256)):
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize(target_size)  # Resize to target size (256, 256)
    image_array = np.array(image) / 255.0  # Normalize image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Define image segmentation route
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Ensure the uploaded file is a .png image
    if not image.filename.endswith(".png"):
        return {"message": "Please upload a .png image."}

    # Read the image
    image_bytes = await image.read()
    image = PILImage.open(BytesIO(image_bytes))  # Open the image from byte data
    
    # Preprocess the image for model prediction
    image_array = preprocess_image(image)
    
    # Make a prediction using the model
    prediction = model.predict(image_array)  # Model prediction
    
    # Convert the prediction to an image
    pred_image = (prediction[0] > 0.5).astype(np.uint8)  # Binarize the prediction
    
    # Convert numpy array to image for response
    pred_image_pil = PILImage.fromarray(pred_image[0] * 255)  # Convert to 0-255 scale
    
    # Save the prediction image to a BytesIO object to return it as a file
    pred_image_bytes = BytesIO()
    pred_image_pil.save(pred_image_bytes, format="PNG")
    pred_image_bytes.seek(0)  # Reset pointer
    
    # Return the predicted image as a StreamingResponse (this will send the image file back)
    return StreamingResponse(pred_image_bytes, media_type="image/png")

# Get the IP address dynamically (for localhost or external IP)
host = socket.gethostbyname(socket.gethostname())  # Get local IP address
port = 8000  # Default port for FastAPI

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)
