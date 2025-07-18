# Legal Entity Recognition with GLiner

This project demonstrates a pipeline for processing legal documents and extracting named entities using the GLiner model.

## Overview

The process is divided into two main parts:

1.  **Data Preprocessing (`preprocessing.py`)**: This script prepares the raw legal text data for the model.
2.  **Inference (`gliner_inference.py`)**: This script uses a pre-trained GLiner model to identify and extract entities from the preprocessed data.

---

### Data Preprocessing (`preprocessing.py`)

This script is responsible for taking the raw data and transforming it into a format suitable for the entity recognition model.

#### Key Functions

*   **Dataset**: The raw data is sourced from the `Hugging Face` dataset `nlpaueb/legal-contracts`(https://huggingface.co/datasets/albertvillanova/legal_contracts/blob/main/legal_contracts.py). A sample of this dataset is provided in `data/raw/legal_contracts_sample.jsonl.gz`.
*   **Sentence Segmentation**: The script uses the `spaCy` library to intelligently split the legal text into individual sentences. This is a crucial step to ensure that the meaning and context of the text are preserved.
*   **Chunking Strategy**:
    *   Legal documents can be very long. To handle this, the script splits the text into smaller, manageable chunks.
    *   An **overlapping** strategy is employed. Each chunk shares some amount of text with the chunk that came before it. This gives the model more context, especially for entities that might appear at the very beginning or end of a chunk.
    *   The script also uses an **offsetting** technique to keep track of where each chunk came from in the original document. This is important for mapping the extracted entities back to their original positions.

---

### GLiner Inference (`gliner_inference.py`)

This script takes the preprocessed data and uses the GLiner model to perform named entity recognition.

#### Key Functions

*   **Model**: The script loads a pre-trained `GLiner` model from `Hugging Face`: `urchade/gliner_base` (https://huggingface.co/xomad/gliner-model-merge-large-v1.0). This model has been trained to recognize a variety of entities and can be adapted for specific use cases.
*   **Inference**:
    *   The script reads the processed data (the text chunks).
    *   For each chunk, it uses the `gliner.predict_entities` function to identify and label entities based on a predefined set of labels (e.g., "person", "organization", "date").

This entire process allows for the efficient and accurate extraction of valuable information from large and complex legal documents.