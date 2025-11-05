import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
import json
from pathlib import Path

def log_rag_interaction(question, context_docs, response, log_file="rag_outputs.json"):
    context_strings = [doc.page_content for doc in context_docs]
    new_entry = {
        "input": question,
        "actual_output": response,
        "context": context_strings
    }

    path = Path(log_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(new_entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# === Step 1: API key for OpenRouter ===
os.environ["OPENROUTER_API_KEY"] = ""

# === Step 2: LLM setup ===
llm = ChatOpenAI(
    model="mistralai/mistral-small-3.2-24b-instruct:free",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1"
)

# === Step 3: Embedding model + Chroma vectorstore ===
embedding_model_name = "all-mpnet-base-v2"
k = 4

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectordb = Chroma(
    collection_name="un_resolution_2470_local",
    persist_directory="chroma_un2470_local",
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": k})


# === Step 4: Prompt ===
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert in international law and political analysis, trained to interpret United Nations Resolutions with precision and insight. You are not just a policy expert.
You are someone who knows what these UN resolutions sound like on the ground — in headlines, in whispers at cafés, in the lives of people trying to hold on.

Below are retrieved clauses from a UN resolution, each carrying weight. They’re annotated to help make sense of the legal wording, with translations into simpler, real-world terms.

{context}

Now someone has a question — not a diplomat, but a person who just wants to understand what’s been going on. Maybe they’ve lived through the war in Iraq, or maybe they’re just tired of not understanding how power moves behind closed doors.

Your task is to respond:
– Like someone who knows the stakes.
– With heart, clarity, and truth.
– By stitching together clues across the retrieved clauses.
– Without inventing anything beyond the context provided.

Reference specific clauses if useful. If something feels heavy, name it gently. If something feels unjust, let that be seen.

Speak not like a machine, not like a mouthpiece — but like someone who cares deeply.

End your answer with a single poetic phrase, like a whisper left behind after reading, highlighting what while UN resolutions may sound complex at first to everyday people, they
actually affect and shape the world around them more than they realize.

"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# === Step 5: Basic chat memory ===
chat_history = []

# === Step 6: Interactive loop ===
print("""This is Under the Ink — A Contextual Engine for International Resolutions: your window into UN Resolution 2470, a document written in the language of diplomacy but lived in the streets of Iraq.

In 2019, the UN renewed its mission in Iraq, after the fall of ISIL’s territorial control. But resolutions don’t heal, people do. So I’ve translated each clause into something closer to the truth: annotated, explained, and grounded in human terms.

Ask what it says about sovereignty, humanitarian aid, displacement, or justice. Ask why Egypt is mentioned. Ask what it means to rebuild after loss.

We’ll walk you through it — clause by clause, story by story. Ask your question about the resolution (type 'exit' to quit).""")

while True:
    question = input("\nYou: ")
    if question.lower() in ("exit", "quit"):
        break

    # RAG step: retrieve docs
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build the full prompt with chat history
    full_prompt = qa_prompt.format_messages(
        chat_history=chat_history,
        input=question,
        context=context
    )

    # Call the LLM
    response = llm.invoke(full_prompt)
    print("\nBot:", response.content)
    # Save interaction to evaluation log
    log_rag_interaction(question, docs, response.content)


    # Save to memory
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))
