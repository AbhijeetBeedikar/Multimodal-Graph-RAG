# Multimodal Enterprise RAG

A multimodal Retrieval-Augmented Generation (RAG) system for enterprise applications.

## Prerequisites

- Python 3.8+
- Google Colab (for Option 1)
- Required Python packages (installed via setup)

## Setup

### Configuration

Before running the code, you **must** update the API key:

1. Navigate to `env.py`
2. Replace the placeholder API key with your actual API key
3. **Important**: The current key in `env.py` is a dummy placeholder and will not work

### Directory Structure

The root folder must be `multimodal_enterprise_rag`. 

**Note**: In the provided code, the folder path is `/content/drive/MyDrive/AI_Projects/multimodal_enterprise_rag` because it was designed to run on Google Colab with files stored in Google Drive. Adjust this path according to your environment.

## Running the Code

### Option 1: Run the Colab Notebook

This is the recommended approach if you're using Google Colab.

1. Navigate to the `rag_system` folder
2. Open `Copy of Enterprise_RAG.ipynb`
3. Run all code blocks in sequential order
4. Provide user input using the function format:
   ```python
   user_input(media_path, query)
   ```

### Option 2: Use the Python Files Directly

This option allows you to use the RAG system as a Python package.

#### Step 1: Install the package

```python
import os
os.chdir("/root")
!pip install -e .
```

#### Step 2: Configure the Python path

```python
import os
import sys
os.chdir("/root")
sys.path.append("/root")
```

#### Step 3: Import and run

```python
import importlib 
import rag_system.main as main 
importlib.reload(main) 
main.user_input(media_path, query)
```

## Usage

The main interface is through the `user_input()` function:

```python
user_input(media_path, query)
```

**Parameters:**
- `media_path`: Path to the media file(s) you want to process
- `query`: Your question or query string

## Troubleshooting

- **Import errors**: Ensure you've run the setup commands in the correct order
- **API errors**: Verify your API key in `env.py` is valid and active
- **Path errors**: Confirm the root directory matches your environment setup

## Project Structure

```
multimodal_enterprise_rag/
│
├── rag_system/
│   ├── ingestion/
│   ├── vector_db/
│   ├── graph/
│   ├── retrieval/
│   ├── main.py
│   ├── env.py
│   └── Copy of Enterprise_RAG.ipynb
│
├── setup.py
├── requirements.txt
└── README.md
```

## Support

For issues or questions, please refer to the project documentation or open an issue in the repository.
