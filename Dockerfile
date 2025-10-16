# -------- Base image --------
FROM python:3.11-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini curl \
    ffmpeg \
    libreoffice-writer libreoffice-calc fonts-dejavu \
    imagemagick libheif-examples libheif1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 10001 appuser
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy only your application code (keep context small via .dockerignore)
COPY app/ ./app/

# Drop privileges
USER appuser

# tini -> uvicorn (production; workers scale via --workers if needed)
CMD ["uvicorn","app.main:app","--host=0.0.0.0","--port=8000"]

