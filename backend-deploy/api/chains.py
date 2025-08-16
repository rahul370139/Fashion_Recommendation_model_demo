from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
# from langchain.tools import tool
# import requests, os

# SEARCH_API = os.getenv("MW_API_ROOT", "http://localhost:8000")

# @tool("fashion_search", return_direct=True)
# def fashion_search(image_path: str, query: str = "") -> str:
#     """Return JSON search results."""
#     files = {"file": open(image_path, "rb")}
#     data = {"text": query}
#     r = requests.post(f"{SEARCH_API}/search", files=files, data=data, timeout=30)
#     return r.text

prompt = ChatPromptTemplate.from_template(
    "You are a friendly fashion stylist.\nUser: {input}\nAssistant:"
)
llm = OllamaLLM(model="mistral")
stylist_chain = prompt | llm  # simple LLMChain for text-only chat

async def chat_with_stylist(query: str) -> str:
    """Async function to chat with the stylist"""
    try:
        result = await stylist_chain.ainvoke({"input": query})
        return result.content if hasattr(result, 'content') else str(result)
    except Exception as e:
        print(f"Error in chat: {e}")
        return f"Sorry, I'm having trouble responding right now. Error: {str(e)}" 