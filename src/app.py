import gradio as gr
from generate import answer


def respond(question):
    if not question.strip():
        return "Please enter a question.", ""
    result = answer(question, rerank=True, hybrid=True)
    sources = "\n".join(
        f"[{i}] {s['source']} p.{s['page']}" for i, s in enumerate(result["sources"], 1)
    )
    return result["answer"], sources


demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Question", lines=2),
    outputs=[gr.Textbox(label="Answer"), gr.Textbox(label="Sources (paper, page)")],
    title="RAG over arXiv papers",
    description="Hybrid search + reranker + local LLM, with citations. Answers only from the indexed papers.",
    examples=[
        ["How do rerankers improve retrieval quality?"],
        ["Why can't we guarantee that LLMs will use useful retrieved information?"],
        ["How do I train a dog to sit?"],
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()