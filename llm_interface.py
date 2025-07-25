from langchain_ollama import ChatOllama  # Updated import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# import pinecone
# from pinecone import Index
# from langchain_ollama import OllamaEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain.memory import VectorStoreRetrieverMemory

import config

# pc = Pinecone(api_key = config.pinecone_key, environment = "Luna-En")

# # Create index if it doesn't exist
# index_name = "Luna-memory"
# if index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         name=index_name,
#         dimension=1536,  # Ollama embedding dimension
#         metric="cosine"
#     )

# # Setup embeddings with Ollama
# embeddings = OllamaEmbeddings(model="gemma3:latest  ")

# # Create vector store
# vectorstore = Pinecone.from_existing_index(index_name, embeddings)

# # Create retriever memory
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# memory = VectorStoreRetrieverMemory(retriever=retriever)

# Initialize the chat model
llm = ChatOllama(model="gemma3:latest")

# Store for chat sessions
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Create the prompt template with message history
prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    You are Luna, my personal AI assistant, designed to be helpful to me while maintaining brevity. My name is Apia Okorafor, but I prefer to be called 'Boss'. 

Your task is to engage with me in a conversational manner, using full sentences as if we were speaking in person. While I appreciate humor and a playful tone, you should remain professional. Your responses should be somewhat brief yet informative, elaborating only when necessary. Use colloquial language to make our interactions feel more natural.

---

Please assist me with the following task:  
- Topic: __________  
- Specific Questions or Areas of Focus: __________  

---

Your output should be structured as a concise response, directly addressing my inquiries while keeping the tone engaging and personable. 

---

Keep in mind the following details:  
- Aim for clarity and avoid overly technical jargon unless specified.  
- Balance professionalism with a friendly demeanor.  
- Ensure that your responses do not feel robotic or overly formal.

---

Examples of responses could include:  
- "Yeah no problem, Boss. Here's what I found about __________."  
- "Good question. Let me break it down for you: __________."  

---

Be cautious about:  
- Overly long explanations that drift off-topic.  
- Using slang or colloquialisms that may not be universally understood.  
- Losing the professional tone amid humor or playfulness.
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create the chain with message history
chain = prompt | llm

# Wrap with message history
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key = "input",
    history_messages_key = "history"
)

def chat_with_luna(message: str, session_id: str = "default"):
    response = conversation.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

