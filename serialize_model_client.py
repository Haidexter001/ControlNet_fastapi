import requests
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Request model serialization via FastAPI.")
parser.add_argument("--config_path", required=True, help="Path to the YAML configuration file.")
parser.add_argument("--state_dict_path", required=True, help="Path to the .pth file containing the model state dictionary.")
parser.add_argument("--url", required=False, default="http://127.0.0.1:8000/serialize/", help="URL for the FastAPI serialization endpoint. Defaults to 'http://127.0.0.1:8000/serialize/'.")
args = parser.parse_args()

# Prepare the JSON request body
data = {
    "config_path": args.config_path,
    "state_dict_path": args.state_dict_path,
}
# Send the POST request
try:
    response = requests.post(args.url, json=data)

    # Check the response status
    if response.status_code == 200:
        print(f"Success! {response.json()}")
    else:
        print(f"Failed to serialize model. Status code: {response.status_code}, Response: {response.text}")
except Exception as e:
    print(f"An error occurred: {e}")
