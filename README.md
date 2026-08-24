# Ai-Smart-asistence
this project is for my college which comes under the minor project 
The application was initially developed and tested successfully in a local environment using Ollama and Llama 3.2:3b.

During deployment to Streamlit Cloud, several environment differences were encountered. The project was updated to support the deployment environment, including reorganizing the utils package, updating dependencies, handling FAISS/PyTorch/TorchVision requirements, and modifying the Ollama connection from a local endpoint to a cloud-accessible endpoint.
file are :
── rag_pipeline.py
├── question_generator.py
└── summarizer.py
