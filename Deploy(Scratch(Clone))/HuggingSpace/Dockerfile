# Base image
FROM python:3.9

# Buat user non-root
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy source code aplikasi dan resource
COPY --chown=user Interface /app/Interface
COPY --chown=user Models /app/Models
COPY --chown=user Reports_csv /app/Reports_csv
COPY --chown=user Images /app/Images
COPY --chown=user README.md structure.txt .gitattributes /app/

# Jalankan Streamlit app
CMD ["streamlit", "run", "Interface/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]