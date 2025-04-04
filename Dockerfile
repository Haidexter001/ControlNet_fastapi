FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the required folders and scripts to the container
COPY ./main.py /app/main.py
COPY ./model.py /app/model.py
COPY ./saved_model /app/saved_model
COPY ./annotator /app/annotator
COPY ./cldm /app/cldm
COPY ./ldm /app/ldm

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]