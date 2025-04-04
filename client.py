import requests
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Send an image to FastAPI for inference and save the output.")
parser.add_argument("--image", required=True, help="Path to the input image file.")
parser.add_argument("--save", required=True, help="Path to save the generated output image.")
args = parser.parse_args()

# Get the input and output file paths from command-line arguments
input_image_path = args.image
output_image_path = args.save

# Endpoint URL
url = "http://127.0.0.1:8000/generate/"  # Update with your FastAPI endpoint if necessary

# Parameters for the FastAPI endpoint
data = {
    "prompt": "mri brain scan",
    "a_prompt": "good quality",
    "n_prompt": "animal, drawing, painting, vivid colors, lowres, bad anatomy",
    "num_samples": 1,
    "image_resolution": 512,
    "ddim_steps": 10,
    "guess_mode": False,
    "strength": 1.0,
    "scale": 9.0,
    "seed": 1,
    "eta": 0.0,
    "low_threshold": 50,
    "high_threshold": 100,
}

try:
    # Serialize the image and send the request
    with open(input_image_path, "rb") as f:
        files = {"file": f}  # Attach the image file
        response = requests.post(url, files=files, data=data)

    # Check the response status
    if response.status_code == 200:
        # Save the response image
        with open(output_image_path, "wb") as f:
            f.write(response.content)
        print(f"Generated image saved at {output_image_path}")
    else:
        print(f"Failed to generate image. Status code: {response.status_code}, Response: {response.text}")
except Exception as e:
    print(f"An error occurred: {e}")
