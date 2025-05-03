# Multi-Task NLP Gradio Demo

This project is an interactive web application demonstrating several Natural Language Processing (NLP) tasks using Hugging Face `transformers` and Gradio.

The application is built using the `pipeline` feature from the Hugging Face `transformers` library, allowing users to easily interact with powerful pre-trained models for different tasks via a user-friendly interface built with Gradio.

## Features

The application currently includes the following NLP tasks, each in its own tab:

1.  **Sentiment Analysis:** Determines the sentiment (Positive/Negative) of English text.
    * Model: [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

2.  **Text Summarization:** Provides a concise summary of longer English text.
    * Model: [`sshleifer/distilbart-cnn-12-6`](https://huggingcerningface.co/sshleifer/distilbart-cnn-12-6)

3.  **Named Entity Recognition (NER):** Identifies and highlights named entities (like Persons, Organizations, Locations) in English text.
    * Model: [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER)

## Technologies Used

* [Python](https://www.python.org/)
* [Hugging Face `transformers`](https://huggingface.co/docs/transformers/index)
* [PyTorch](https://pytorch.org/) (or TensorFlow, depending on your installation, but PyTorch is common with Gradio examples)
* [Gradio](https://www.gradio.app/)

## Setup and Running Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/fxs2596/multi-task-nlp-gradio-demo.git](https://github.com/fxs2596/multi-task-nlp-gradio-demo.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd multi-task-nlp-gradio-demo
    ```
3.  **Install dependencies:** Make sure you have Python and pip installed. Then install the required libraries.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If `pip install -r requirements.txt` gives errors, you might need to install `torch` or `tensorflow` separately first depending on your system, e.g., `pip install torch`)*
4.  **Run the application:**
    ```bash
    python gradiooo.py
    ```
5.  The application will start a local web server and provide a URL (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser.
6.  If you used `share=True` in the script, it will also provide a temporary public URL you can share.

## Deployment

This application can be easily deployed to platforms that support Gradio, such as [Hugging Face Spaces](https://huggingface.co/spaces). Hugging Face Spaces provides free hosting for Gradio demos, often directly from a GitHub repository.

*(Link to live demo on Hugging Face Spaces - add this here after deployment)*

## Screenshot

![image](https://github.com/user-attachments/assets/d7c570f2-d9a5-4d67-a97f-42061fb0b77c)


---

Feel free to explore the code and the different NLP tasks!
