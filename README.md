# Log Classification With Hybrid Classification Framework

This project implements a hybrid log classification system that combines three complementary approaches to handle log patterns of different complexity levels. It uses Regex for simple rule-based cases, Sentence Transformer plus Logistic Regression for learned semantic classification, and an LLM-based fallback for more ambiguous cases where labeled data may be limited. The project is now also available through a live Streamlit web app for interactive testing and CSV-based output generation.

## Live Demo

- **Streamlit App:** [Log Classification NLP BERT LLM](https://log-classification-nlp-bert-llm-dp.streamlit.app/)
- **GitHub Repository:** [Log_Classification_NLP_BERT_LLM](https://github.com/dipanshuparashar902/Log_Classification_NLP_BERT_LLM)

![architecture](resources/arch.png)

## Classification Approaches

### 1. Regular Expression (Regex)
- Handles simple, predictable log patterns.
- Useful when log messages follow clearly defined rules.
- Provides a fast rule-based baseline for deterministic classifications.

### 2. Sentence Transformer + Logistic Regression
- Handles more complex patterns when sufficient training data is available.
- Generates embeddings using Sentence Transformers.
- Uses Logistic Regression as the classification layer.
- Works well for semantically similar log messages even when exact wording changes.

### 3. LLM (Large Language Model)
- Handles complex or less structured log messages.
- Useful when sufficient labeled examples are not available.
- Acts as a fallback or complementary classifier in the hybrid system.

## Features

- Hybrid classification pipeline using Regex, BERT-style sentence embeddings, and LLMs
- Streamlit web app for file upload and interactive classification
- CSV output generation with downloadable results
- Optional email sharing of the output file from the app
- Modular project structure for training, inference, and UI deployment

## Folder Structure

```text
Log_Classification_NLP_BERT_LLM/
│── models/                  # Saved model artifacts
│── resources/               # Test files, outputs, architecture image, etc.
│── training/                # Model training scripts
│── processor_bert.py        # BERT-based classification logic
│── processor_llm.py         # LLM-based classification logic
│── processor_regex.py       # Regex-based classification logic
│── streamlit_app.py         # Streamlit user interface
│── requirements.txt         # Project dependencies
│── README.md                # Project documentation
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/dipanshuparashar902/Log_Classification_NLP_BERT_LLM.git
cd Log_Classification_NLP_BERT_LLM
```

### 2. Create and Activate Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root if you want to use the LLM or email functionality:

```env
GROQ_API_KEY=your_groq_api_key
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=465
```

## Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

After running the app, open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Usage

1. Upload a CSV file containing a `log_message` column, or upload a TXT file with one log message per line.
2. Select a classification method: Regex, BERT, or LLM.
3. Review the results in the interactive table.
4. Download the output as a CSV file.
5. Optionally send the output file via email from the app.

## Output

The application generates structured output with fields such as:
- `log_message`
- `predicted_label`
- `classifier_used`
- `status`

## Tech Stack

- **Frontend / UI:** Streamlit
- **Embedding Model:** Sentence Transformers
- **ML Classifier:** Logistic Regression
- **Rule Engine:** Regex
- **LLM Integration:** Groq API
- **Data Handling:** Pandas

## Future Improvements

- Add confidence scores for predictions
- Support multi-class and hierarchical log categorization
- Add model comparison dashboard inside Streamlit
- Improve audit logging and alert workflows
- Extend deployment to enterprise monitoring pipelines

