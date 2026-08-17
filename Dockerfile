FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies for MobileNetV2 / OpenCV if needed (often required by cv2)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY . .

# Expose port 5000
EXPOSE 5000

# Run the Flask app directly to save memory on 512MB Render instance
CMD ["python", "app.py"]
