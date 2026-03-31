FROM python:3.12-slim

# Install system dependencies for OCR and PDF handling safely
# We clear out apt-lists instantly to drastically reduce memory usage during builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory inside the container
WORKDIR /app

# Copy the exact requirements over first
COPY requirements.txt .

# Install dependencies strictly with no cache to prevent RAM exhaustion during pip install
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend into the container
COPY . .

# Expose port (Render automatically sets random PORT env variable, defaulting to 5000 locally)
EXPOSE 5000

# Start server identically to how Waitress works in app.py when not locally debugging
CMD ["python", "app.py"]
