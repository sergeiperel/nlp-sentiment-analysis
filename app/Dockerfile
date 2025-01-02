# Base image -- Python
FROM python:3.11-slim

# Working directory in our container
WORKDIR /app

# Copy required files
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY api.py api.py

# Install dependencies
RUN apt-get update && apt-get install -y g++
RUN pip install -r requirements.txt
# TODO: nltk download etc. for preprocessing

# Run when container starts
CMD streamlit run app.py
