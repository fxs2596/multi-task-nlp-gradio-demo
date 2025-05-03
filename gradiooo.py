# gradiooo.py
# This script creates a polished Gradio web interface with multiple NLP tasks
# using Hugging Face pipelines (Sentiment, Summarization, NER).
#
# Prerequisites: pip install transformers torch gradio

from transformers import pipeline
import gradio as gr
import torch
import gradio.themes as gr_themes

# --- Model names & Hugging Face URLs ---
sentiment_model_name     = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
summarization_model_name = "sshleifer/distilbart-cnn-12-6"
ner_model_name           = "dslim/bert-base-NER"

sentiment_model_url     = f"https://huggingface.co/{sentiment_model_name}"
summarization_model_url = f"https://huggingface.co/{summarization_model_name}"
ner_model_url           = f"https://huggingface.co/{ner_model_name}"

# --- Load Hugging Face pipelines ---
print("Loading NLP pipelines…")
sentiment_classifier = pipeline("sentiment-analysis", model=sentiment_model_name)
print(f"  • Sentiment pipeline on device: {sentiment_classifier.device}")
summarizer = pipeline("summarization", model=summarization_model_name)
print(f"  • Summarization pipeline on device: {summarizer.device}")
ner_recognizer = pipeline("ner", model=ner_model_name, aggregation_strategy="simple")
print(f"  • NER pipeline on device: {ner_recognizer.device}")

# --- Prediction functions ---

def get_sentiment_prediction(text):
    """Return {label: score} from the sentiment pipeline."""
    if not text:
        return {}
    result = sentiment_classifier(text)[0]
    return { result["label"]: result["score"] }

def get_summarization_prediction(text):
    """Return a summary string for input text, or an instruction if too short."""
    if not text:
        return ""
    words = text.split()
    if len(words) < 15:
        return "Please provide at least 15 words for summarization."
    # set lengths
    min_len = min(30, len(words) // 4)
    max_len = min(len(words), max(min_len + 20, 120))
    summary = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        early_stopping=True
    )[0]["summary_text"]
    return summary

def format_for_highlight(text, entities):
    """
    Convert HF NER output (list of dicts) to list of (span, label) tuples
    for gr.HighlightedText.
    """
    highlights = []
    last_end = 0
    for ent in entities:
        start, end = ent["start"], ent["end"]
        # non-entity text before this entity
        if start > last_end:
            highlights.append((text[last_end:start], None))
        # the entity itself
        highlights.append((text[start:end], ent["entity_group"]))
        last_end = end
    # trailing text after last entity
    if last_end < len(text):
        highlights.append((text[last_end:], None))
    return highlights

def get_ner_prediction(text):
    """Return list of (span, label) tuples for HighlightedText."""
    if not text:
        return []
    raw = ner_recognizer(text)
    return format_for_highlight(text, raw)

# --- Build the Gradio interface ---

with gr.Blocks(theme=gr_themes.Soft()) as demo:
    gr.Markdown("# Hugging Face Multi‑Task NLP Demo")
    gr.Markdown("Explore Sentiment Analysis, Summarization, and NER via Hugging Face pipelines.")

    # Sentiment Analysis tab
    with gr.Tab("Sentiment Analysis"):
        gr.Markdown(f"""
        **Model:** [{sentiment_model_name}]({sentiment_model_url})  
        Enter English text below to get a sentiment prediction (POS/NEG).
        """)
        sentiment_input = gr.Textbox(
            lines=3,
            placeholder="Type text here…",
            label="Input Text"
        )
        sentiment_output = gr.Label(label="Sentiment")
        gr.Examples(
            examples=[
                ["I loved this movie, it was fantastic!"],
                ["The service was awful and the food was cold."]
            ],
            inputs=sentiment_input
        )
        gr.Button("Analyze Sentiment").click(
            fn=get_sentiment_prediction,
            inputs=sentiment_input,
            outputs=sentiment_output
        )

    # Text Summarization tab
    with gr.Tab("Text Summarization"):
        gr.Markdown(f"""
        **Model:** [{summarization_model_name}]({summarization_model_url})  
        Enter a longer English paragraph (≥15 words) to get a concise summary.
        """)
        summarize_input = gr.Textbox(
            lines=5,
            placeholder="Paste a long text here…",
            label="Input Text"
        )
        summarize_output = gr.Textbox(
            lines=3,
            label="Summary",
            interactive=False
        )
        gr.Examples(
            examples=[
                ["Machine learning is a subset of artificial intelligence that focuses on enabling computers to learn from data and improve over time without being explicitly programmed to perform specific tasks. This approach allows systems to adapt and provide insights based on patterns they detect."],
                ["The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet and is often used to test fonts or keyboards."]
            ],
            inputs=summarize_input
        )
        gr.Button("Summarize Text").click(
            fn=get_summarization_prediction,
            inputs=summarize_input,
            outputs=summarize_output
        )

    # Named Entity Recognition tab
    with gr.Tab("Named Entity Recognition"):
        gr.Markdown(f"""
        **Model:** [{ner_model_name}]({ner_model_url})  
        Enter English text below to highlight named entities.
        """)
        ner_input = gr.Textbox(
            lines=3,
            placeholder="Type text here…",
            label="Input Text"
        )
        ner_output = gr.HighlightedText(label="Entities")
        gr.Examples(
            examples=[
                ["Barack Obama visited Berlin on Monday."],
                ["Apple Inc. announced new products at WWDC in San Jose."]
            ],
            inputs=ner_input
        )
        gr.Button("Recognize Entities").click(
            fn=get_ner_prediction,
            inputs=ner_input,
            outputs=ner_output
        )

# Launch the interface
demo.launch(share=True)
