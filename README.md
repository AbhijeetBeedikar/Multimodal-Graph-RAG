# Multimodal-Graph-RAG

A multimodal Retrieval-Augmented Generation (RAG) system that combines vector search with knowledge graph retrieval for enhanced enterprise applications.

## Prerequisites

- Python 3.8+
- Required Python packages (installed via setup)
- Gemini API key for the underlying language model

## Setup

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Multimodal-Graph-RAG.git
cd Multimodal-Graph-RAG
```

2. Install the package and dependencies:
```bash
pip install -e .
```

This will automatically install all required packages listed in `requirements.txt`.

### Configuration

The application uses Streamlit secrets management for secure API key storage:

1. Create a `.streamlit` folder in the project root (if it doesn't exist)
2. Create a `secrets.toml` file inside `.streamlit`:
```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

3. Add your API key to `secrets.toml`:
```toml
API_KEY = "your-actual-api-key-here"
```

4. **Important**: Never commit `secrets.toml` to version control. It's already included in `.gitignore`.

### Directory Structure

The root folder must be `Multimodal-Graph-RAG`. The project is organized as follows:

```
Multimodal-Graph-RAG/
│
├── .devcontainer/          # Development container configuration
├── .idea/                  # IDE configuration files
├── qdrant_storage/         # Vector database storage
│
├── rag_system/             # Core RAG system modules
│   ├── evaluation/
│   │   └── deep_eval.py    # Evaluation metrics (under construction)
│   ├── graph/
│   │   ├── graph_builder.py    # Knowledge graph construction
│   │   ├── graph_functions.py  # Graph utility functions
│   │   └── graph_search.py     # Graph-based retrieval
│   ├── ingestion/
│   │   ├── audio_ingest.py     # Audio file processing
│   │   ├── env.py              # Returns API key
│   │   ├── img_ingest.py       # Image file processing
│   │   └── pdfingestion.py     # PDF document processing
│   ├── rag/
│   │   └── rag_pipeline.py     # Main RAG orchestration
│   ├── retrieval/
│   │   └── orchestrator.py     # Retrieval coordination
│   └── vector_db/
│       └── indexing.py         # Vector database indexing
│
├── .streamlit/             # Streamlit configuration (create locally)
│   └── secrets.toml        # API keys (DO NOT COMMIT)
│
├── LICENSE                 # MIT License
├── README.md              # This file
├── knowledge_graph.gpickle # Serialized knowledge graph
├── main.py                # Streamlit web application
├── requirements.txt       # Python dependencies
└── setup.py              # Package installation configuration
```

## Running the Application

### Streamlit Web Interface (Recommended)

The application is hosted as a Streamlit web app, providing an intuitive interface for interacting with the RAG system.

1. Start the Streamlit server:
```bash
streamlit run main.py
```

2. Your browser should automatically open to `http://localhost:8501`

3. Upload your media files (images, PDFs, audio) and enter your query through the web interface


## Features

- **Multimodal Input Support**: Process images, PDFs, and audio files
- **Hybrid Retrieval**: Combines vector similarity search with knowledge graph traversal
- **Web Interface**: Easy-to-use Streamlit interface for non-technical users
- **Secure Configuration**: API keys managed through Streamlit secrets
- **Knowledge Graph Integration**: Enhanced retrieval through graph-based relationships

## Usage Example

### Through the Web Interface

1. Launch the app with `streamlit run main.py`
2. Upload your media files using the file uploader
3. Enter your query in the text input field
4. Click "Submit" to get your answer


## Troubleshooting

### API Key Issues
- **Error: "API key not found"**: Ensure `secrets.toml` exists in `.streamlit/` and contains your API key
- **Error: "Invalid API key"**: Verify your API key is active and correctly copied

### Path Issues
- Ensure you're running commands from the `Multimodal-Graph-RAG` root directory
- Use absolute paths when specifying media folders programmatically

### Import Errors
- Run `pip install -e .` to ensure all dependencies are installed
- Check that your Python version is 3.8 or higher: `python --version`

### Streamlit Issues
- If the browser doesn't open automatically, manually navigate to `http://localhost:8501`
- To run on a different port: `streamlit run main.py --server.port 8502`

## System Architecture

The RAG system combines four key components:

1. **Ingestion Layer**: Processes multimodal inputs (audio, images, PDFs) into structured formats
2. **Storage Layer**: Maintains both vector embeddings (Qdrant) and knowledge graph (NetworkX)
3. **Retrieval Layer**: Orchestrates hybrid search across vector and graph stores
4. **Generation Layer**: Synthesizes retrieved information into coherent responses

## Performance Notes

- First-time ingestion of large document sets may take several minutes
- The knowledge graph is persisted to `knowledge_graph.gpickle` for faster subsequent loads
- Vector embeddings are stored in `qdrant_storage/` and reused across sessions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues or questions:
- Open an issue in the GitHub repository
- Check existing issues for similar problems
- Refer to the inline documentation in the codebase

## Roadmap

- [ ] Complete evaluation metrics implementation (`deep_eval.py`)
- [ ] Add support for additional media types (video, spreadsheets)
- [ ] Implement caching for faster repeated queries
- [ ] Add batch processing capabilities
- [ ] Enhanced visualization of knowledge graph relationships