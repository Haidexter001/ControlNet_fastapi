import numpy as np
import cv2
import torch

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from io import BytesIO
from PIL import Image
from model import process
from pydantic import BaseModel
from cldm.model import create_model, load_state_dict

app = FastAPI()

# Define a request model for serialize_model endpoint
class SerializeModelRequest(BaseModel):
    config_path: str
    state_dict_path: str

save_path: str = "saved_model/frozen_model.pt"

# Initialize FastAPI app
app = FastAPI()

@app.post("/serialize/")
async def serialize_model(request: SerializeModelRequest):
    """
    Endpoint to initialize and serialize the model based on provided config and state dict paths.
    """
    try:
        # Initialize the model using the provided paths
        model = create_model(request.config_path).cpu()  # Create the model architecture
        model.load_state_dict(load_state_dict(request.state_dict_path, location="cpu"))  # Load weights

        # Serialize and save the model
        torch.save(model, save_path)

        return {"message": "Model serialized successfully!", "save_path": request.save_path}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate/")
async def generate_image(
    save_path: str,
    file: UploadFile = File(...),
    prompt: str = "mri brain scan",
    a_prompt: str = "good quality",
    n_prompt: str = "animal, drawing, painting, vivid colors, lowres, bad anatomy",
    num_samples: int = 1,
    image_resolution: int = 512,
    ddim_steps: int = 10,
    guess_mode: bool = False,
    strength: float = 1.0,
    scale: float = 9.0,
    seed: int = 1,
    eta: float = 0.0,
    low_threshold: int = 50,
    high_threshold: int = 100,
):
    # Read the uploaded image
    print("Reading image")
    input_image = Image.open(BytesIO(await file.read()))
    input_image = np.array(input_image)

    # Generate results
    print("Generating new image...")
    results = process(
        input_image=input_image,
        prompt=prompt,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        num_samples=num_samples,
        image_resolution=image_resolution,
        ddim_steps=ddim_steps,
        guess_mode=guess_mode,
        strength=strength,
        scale=scale,
        seed=seed,
        eta=eta,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    # Save the result as a temporary file
    print("Saving image")
    output_path = f"{save_path}/output.png"
    cv2.imwrite(output_path, results[1])  # Save the first generated result

    # Return the file as the response
    return FileResponse(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

