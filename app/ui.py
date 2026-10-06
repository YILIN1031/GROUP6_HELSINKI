import gradio as gr
from src.services import ai_service

EXAMPLE_QUESTIONS = [
    "What kinds of texts should I practise for the writing test?",
    "Which polite phrases can I use to begin and end a formal email in Swedish?",
    "How do I write a polite complaint to a shop about a faulty product?",
    "How can I say in Swedish that something in my flat is broken?",
    "Can you give me a few practice questions to help me prepare for the exam?",
]

INTRO = """
# CourseMate

**Your local study assistant for course materials.**

Ask a question about your indexed course documents. CourseMate searches the
material first and answers from what it finds, listing the documents it used
under **Sources**. If the material does not cover your question, it will say so.

*Setup:* add PDF, TXT, or Markdown course documents, run
`python -m src.rag.ingest`, then (re)start CourseMate. See the README for details.
"""


def respond(message: str) -> str:
    """
    UI callback. Delegates to the service layer and returns display-ready text
    (the answer plus its Sources list, or a friendly message).

    `generate_response` is looked up at call time so the service layer remains
    the single entry point for every user-facing AI operation.
    """
    return ai_service.generate_response(message)


def build_ui() -> gr.Blocks:
    """
    Constructs the Gradio web interface.

    Architectural Principle: The UI communicates strictly with `generate_response()`
    in the AI service layer and never directly with Ollama, the model client,
    embeddings, the vector store, or ingestion code.
    """
    with gr.Blocks(title="CourseMate") as demo:
        gr.Markdown(INTRO)

        with gr.Row():
            user_input = gr.Textbox(
                lines=3,
                placeholder="Ask a question about your course materials...",
                label="Your question",
            )

        submit_btn = gr.Button("Ask CourseMate", variant="primary")

        # Markdown output so answers and the Sources list render as formatted text.
        output_box = gr.Markdown(
            value="",
            label="CourseMate answer",
            show_label=True,
            container=True,
            line_breaks=True,
            sanitize_html=True,
            min_height=120,
        )

        # Examples only fill in the question box; they never run the model on their own.
        gr.Examples(
            examples=EXAMPLE_QUESTIONS,
            inputs=user_input,
            label="Example questions",
            cache_examples=False,
        )

        # Connect UI actions exclusively to the service layer function
        submit_btn.click(
            fn=respond,
            inputs=user_input,
            outputs=output_box,
        )
        user_input.submit(
            fn=respond,
            inputs=user_input,
            outputs=output_box,
        )

    return demo
