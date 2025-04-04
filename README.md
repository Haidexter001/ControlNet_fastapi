# FastAPI ControlNet Image Generation API with Model Serialization

This project provides a FastAPI-based API to handle dynamic model initialization, serialization, and image generation using machine learning models. Users can specify configuration and state dictionary paths to serialize models, and upload images for inference to generate new outputs.

## Features

**Model Serialization:** Dynamically initialize and serialize machine learning models based on user-provided configuration and state dictionary paths.

**Image Generation:** Generate new images based on user-uploaded input and a customizable set of parameters.

**Dockerized Application:** The app is fully containerized for easy deployment and reproducibility.

**Secure Output Handling:** Ensures serialized models are saved in a dedicated directory (saved_model) within the container.

## Project Structure

project-folder/
- main.py
- model.py
- Dockerfile
- requirements.txt
- .dockerignore
- client.py
- serialized_model_client.py
- saved_model/
  - frozen_model.pt
  - ...
- output/
  - ...
- saved_model/
  - frozen_model.pt
- model/
  - cldm_v15.yaml
  - control_sd15_canny.pth
  - ...
- annotator/
  - ...
- cldm/
  - ...
- ldm/
  - ...
    
## Requirements

    Python 3.8

    Docker (for containerization)

    Python dependencies specified in requirements.txt

Install the required Python dependencies:

    pip install -r requirements.txt

## How to Run

### 1. Local Development

Run the application locally:

    python main.py
The FastAPI application starts at http://127.0.0.1:8000.

### 2. Using Docker

Build the Docker Image:

    docker build -t controlnet-app.

Run the Container:

    docker run -p 8000:8000 -v "$(pwd)/saved_model:/app/saved_model" -v "$(pwd)/output:/app/output" controlnet-app

## API Endpoints

### POST /serialize/

Serialize a Pytorch model to .pt format.

- **Request Body:**


    {
        "config_path": "model/cldm_v15.yaml",
        "state_dict_path": "model/control_sd15_canny.pth"
    }

Alternatively, use serialize_model_client.py

    python serialize_client.py --config_path model/cldm_v15.yaml --state_dict_path model/control_sd15_canny.pth

- **Response:**


    {
        "message": "Model serialized successfully!",
        "save_path": "saved_model/frozen_model.pt"
    }
### POST /generate/

Generate a new image from an uploaded input image.
- **Request Parameters:**


    save_path: Path to the output directory.

    file: The input image file to be processed.

    prompt: The primary text prompt for generating the new image (default: "mri brain scan").

    a_prompt: Additional positive attributes for the image (default: "good quality").

    n_prompt: Negative attributes to exclude from the generated image (default: "animal, drawing, painting, vivid colors, lowres, bad anatomy").

    num_samples: Number of images to generate (default: 1).

    image_resolution: Resolution of the generated image (default: 512).

    ddim_steps: Number of DDIM sampling steps (default: 10).

    guess_mode: Boolean flag for guess mode (default: False).

    strength: Strength of control factors (default: 1.0).

    scale: Guidance scale for the image generation process (default: 9.0).

    seed: Seed for random number generation (default: 1).

    eta: Eta for the DDIM sampler (default: 0.0).

    low_threshold: Low threshold for edge detection (default: 50).

    high_threshold: High threshold for edge detection (default: 100).

- **Response:**

Returns the generated image as a file.

## Usage Examples
### Serialize a Model

#### Using curl:

    curl -X POST "http://127.0.0.1:8000/serialize/" \
    -H "Content-Type: application/json" \
    -d '{
        "config_path": "model/cldm_v15.yaml",
        "state_dict_path": "model/control_sd15_canny.pth"
    }'

#### Python Client for Serializing Model
    
    python serialize_model_client.py --config_path <path-to-yaml-config> --state_dict_path <path-to-state-dict>

### Generate an Image

#### Using curl:

    curl -X POST "http://127.0.0.1:8000/generate/" \
    -F "file=@input_image.jpg"
    -F "save_path=saved_model" \

#### Python Client for Image Generation

    python client.py --image <path-to-image> --save <save-directory>

## License

This project is licensed under the MIT License.
