# Use Python as the base image
FROM python:3.14-slim

# Set the working folder inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install the Python packages our app needs
RUN pip install --no-cache-dir \
    streamlit \
    langchain \
    langchain-openai \
    langgraph \
    python-dotenv \
    pypdf \
    pydantic

# Tell Docker that Streamlit uses port 8501
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "web_app.py", "--server.address=0.0.0.0", "--server.port=8501"]