import gradio as gr
from classifier import classify_department
from retriever import retrieve_chunks
from generator import generate_answer


def handle_question(question: str) -> tuple[str, str]:
    question = question.strip()
    if not question:
        return "", "Please enter a customer question."

    try:
        print(f"\n--- New Question ---")
        print(f"Question: {question}")

        department = classify_department(question)
        print(f"Detected department: {department}")

        if department == "unknown":
            print("Could not match to a department — flagging for manual handling.")
            return (
                "Unknown",
                "This question could not be matched to a department. Please handle this manually.",
            )

        chunks = retrieve_chunks(question, department)
        print(f"Chunks retrieved: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {chunk[:80]}...")

        answer = generate_answer(question, chunks)
        print(f"Answer generated: {answer[:100]}...")
        dept_display = department.replace("_", " ").title()
        return dept_display, answer

    except Exception as e:
        print(f"Pipeline error: {e}")
        return "", "An error occurred while processing your question. Please try again."


with gr.Blocks(title="ElectroTech Customer Q&A") as app:
    gr.Markdown("## ElectroTech Customer Q&A Assistant")
    gr.Markdown(
        "Paste a customer email question below. "
        "The system will find the relevant department and suggest a reply."
    )

    question_input = gr.Textbox(
        label="Customer Question",
        placeholder="Paste the customer's question here...",
        lines=4,
    )
    submit_btn = gr.Button("Get Answer", variant="primary")

    with gr.Row():
        dept_output = gr.Textbox(label="Detected Department", interactive=False)
        answer_output = gr.Textbox(label="Suggested Answer", lines=8, interactive=False)

    submit_btn.click(
        fn=handle_question,
        inputs=[question_input],
        outputs=[dept_output, answer_output],
    )

if __name__ == "__main__":
    app.launch()
