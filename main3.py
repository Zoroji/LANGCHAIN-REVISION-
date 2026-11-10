from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
load_dotenv() 

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I think pears taste very good.',
    'I hate bananas.',
    'I dislike raspberries.',
    'I despise mangos.',
    'I love Linux.',
    'I hate Windows.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search('What fruits does the person like?', k=3))
print(vector_store.similarity_search('What fruits does the person hate?', k=3))



retriever = vector_store.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever,
    name="kb_search",
    description="Search the knowledge base for relevant information."
)


agent = create_agent(
    model="groq:qwen/qwen3.8-27b",
    tools=[retriever_tool],
    system_prompt=(
        "You are a helpful assistant. For questions about Macs, apples, or laptops, "
        "first call the kb_search tool to retrieve context, then answer succinctly. "
        "You may need to use it multiple times before answering."
    )
)


result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What three fruits does the person like and what three fruits does the person dislike?"
        }
    ]
})

print(result)
print(result["messages"][-1].content)
