# Use the official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy models
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download es_core_news_sm \
    && python -m spacy download fr_core_news_sm

# Copy the application files
COPY . .

# Expose the port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "main.py"]
