import config
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, List
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory


class PineconeConversationStore:
    # A custom class to store and retrieve conversations using Pinecone vector database.
    # This allows for semantic search and retrieval of similar conversation contexts.
    
    def __init__(self, api_key: str, index_name: str = "luna-conversations"):
        # Initialize the Pinecone conversation store with API key and index name
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Initialize embeddings model (using Ollama's embedding model)
        # This converts text into numerical vectors for semantic similarity
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text:latest"  # You can change this to any embedding model
        )
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        
        # Initialize the vector store
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings
        )
    
    def _create_index_if_not_exists(self):
        # Creates a Pinecone index if it doesn't already exist.
        # An index is like a database table that stores vectors and metadata.
        if self.index_name not in [index["name"] for index in self.pc.list_indexes()]:
            # Create index with specific dimensions (nomic-embed-text uses 768 dimensions)
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Must match your embedding model's output dimensions
                metric="cosine",  # Similarity metric (cosine similarity works well for text)
                spec=ServerlessSpec(
                    cloud="aws",  # or "gcp"
                    region="us-east-1"  # Choose appropriate region
                )
            )
        self.index = self.pc.Index(self.index_name)
    
    def store_conversation_turn(self, session_id: str, human_message: str, ai_response: str, context: Dict[str, Any] = None):
        # Store a single conversation turn (human message + AI response) in Pinecone
        # Create a unique ID for this conversation turn
        turn_id = str(uuid.uuid4())
        
        # Combine human message and AI response for embedding
        # This allows us to search based on either part of the conversation
        conversation_text = f"Apia: {human_message}\nLuna: {ai_response}"
        
        # Prepare metadata to store alongside the vector
        metadata = {
            "session_id": session_id,
            "turn_id": turn_id,
            "human_message": human_message,
            "ai_response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "message_hash": hashlib.md5(conversation_text.encode()).hexdigest()
        }
        
        # Add any additional context
        if context:
            metadata.update(context)
        
        # Create a Document object (LangChain's format for storing text + metadata)
        document = Document(
            page_content=conversation_text,
            metadata=metadata
        )
        
        # Store in Pinecone (this will automatically create embeddings)
        self.vector_store.add_documents([document])
        
        return turn_id
    
    def search_similar_conversations(self, query: str, k: int = 5, session_filter: str = None) -> List[Dict]:
        # Search for conversations similar to the given query
        # Prepare filter for metadata if session_filter is provided
        filter_dict = {"session_id": session_filter} if session_filter else None
        
        # Perform similarity search
        # This converts the query to a vector and finds the most similar stored vectors
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter_dict
        )
        
        # Format results for easy consumption
        formatted_results = []
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_conversation_context(self, session_id: str, query: str, 
                               max_context_turns: int = 3) -> str:
        # Get relevant conversation context for improving AI responses
        # Search for similar conversations across all sessions
        similar_convos = self.search_similar_conversations(
            query=query, 
            k=max_context_turns * 2  # Get more results to filter from
        )
        
        # Prioritize conversations from the same session but include others
        context_parts = []
        same_session_count = 0
        other_session_count = 0
        context_parts.append(f"Current time is {datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")}.")
        
        for convo in similar_convos:
            if len(context_parts) >= max_context_turns:
                break
            
            metadata = convo["metadata"]
            timestamp = datetime.fromisoformat(metadata["timestamp"]).strftime("%A, %B %d, %Y %H:%M:%S")
            
            # Prioritize same session conversations
            if metadata["session_id"] == session_id and same_session_count < max_context_turns // 2:
                context_parts.append(f"[Previous in this session at {timestamp}]: {convo['content']}")
                same_session_count += 1
            elif metadata["session_id"] != session_id and other_session_count < max_context_turns // 2:
                context_parts.append(f"[Similar past conversation at {timestamp}]: {convo['content']}")
                other_session_count += 1
        
        return "\n\n".join(context_parts) if context_parts else ""


# Custom message history class that stores conversations in both memory and Pinecone
class EnhancedChatMessageHistory(BaseChatMessageHistory): # Extends the built in message history class

    # Extends the basic in-memory storage with persistent vector storage
    def __init__(self, session_id: str, pinecone_store: PineconeConversationStore):
        self.session_id = session_id
        self.pinecone_store = pinecone_store
        self.in_memory_history = InMemoryChatMessageHistory()
        self._last_human_message = None

    # Add a message to the history. Required by LangChain's interface.
    def add_message(self, message) -> None:
        # Delegate to the in-memory history
        self.in_memory_history.add_message(message)
        
        # Handle Pinecone storage based on message type
        if hasattr(message, 'type'):
            if message.type == 'human':
                self._last_human_message = message.content
            elif message.type == 'ai' and self._last_human_message:
                # Store the complete conversation turn in Pinecone
                self.pinecone_store.store_conversation_turn(
                    session_id=self.session_id,
                    human_message=self._last_human_message,
                    ai_response=message.content
                )
                self._last_human_message = None

    # Add a user message to the history
    def add_user_message(self, message: str) -> None:
        
        self.in_memory_history.add_user_message(message)
        self._last_human_message = message
    
    # Add an AI message to the history and store the conversation turn in Pinecone
    def add_ai_message(self, message: str) -> None:
        self.in_memory_history.add_ai_message(message)
        # Store the complete conversation turn in Pinecone
        if self._last_human_message:
            self.pinecone_store.store_conversation_turn(
                session_id=self.session_id,
                human_message=self._last_human_message,
                ai_response=message
            )
            self._last_human_message = None
    
    # Add multiple messages to the history. Required by LangChain's interface.
    def add_messages(self, messages) -> None:
        # Delegate to the in-memory history
        self.in_memory_history.add_messages(messages)
        # Process each message for Pinecone storage
        for message in messages:
            if hasattr(message, 'type'):
                if message.type == 'human':
                    self._last_human_message = message.content
                elif message.type == 'ai' and self._last_human_message:
                    self.pinecone_store.store_conversation_turn(
                        session_id=self.session_id,
                        human_message=self._last_human_message,
                        ai_response=message.content
                    )
                    self._last_human_message = None

    # Clear the message history
    def clear(self) -> None:
        self.in_memory_history.clear()
        self._last_human_message = None
    
    # Return the list of messages
    @property
    def messages(self):
        return self.in_memory_history.messages





# Initialize the chat model
llm = ChatOllama(model="gemma3:latest")

pinecone_store = PineconeConversationStore(api_key=config.pinecone_key)

# Store for chat sessions (now uses enhanced history)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    # Get or create a session history with Pinecone integration
    if session_id not in store:
        store[session_id] = EnhancedChatMessageHistory(session_id, pinecone_store)
    return store[session_id]

# Create the prompt template with message history
prompt = ChatPromptTemplate.from_messages([
    ("system", config.system_prompt),
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

# Enhanced chat function with optional context retrieval from Pinecone
def chat_with_luna(message: str, session_id: str = "testing", use_context: bool = True): # Using "testing" as session_id for now

    context = ""
    
    if use_context:
        # Get relevant conversation context from Pinecone
        context = pinecone_store.get_conversation_context(
            session_id=session_id,
            query=message,
            max_context_turns=3
        )
        
        if context:
            context = f"Relevant conversation context:\n{context}\n"
    
    # Invoke the conversation with context
    response = conversation.invoke(
        {
            "input": message,
            "context": context
        },
        config={"configurable": {"session_id": session_id}}
    )
    
    return response.content


# Search through past conversations for specific topics or information
def search_past_conversations(query: str, session_id: str = None, limit: int = 5):
    return pinecone_store.search_similar_conversations(
        query=query,
        k=limit,
        session_filter=session_id
    )

# Example usage function
def test_usage():
    # Demonstrate how to use the enhanced chat system
    print("-" * 50)

    # Regular chat
    response1 = chat_with_luna("Why did the chicken cross the road")
    print("Luna:", response1)
    print("-" * 50)

    # Chat with context disabled
    response2 = chat_with_luna("Tell me a bit about new machine learning advancements", use_context=False)
    print("Luna:", response2)
    print("-" * 50)

    # Regular chat
    response3 = chat_with_luna("What's my name?")
    print("Luna:", response3)

    # Regular chat
    response4 = chat_with_luna("What was the topic of conversation about 5 minutes ago?")
    print("Luna:", response4)
    print("-" * 50)

    # Search past conversations
    past_convos = search_past_conversations("machine learning", limit=3)
    for i, convo in enumerate(past_convos):
        print(f"Past conversation {i+1}:")
        print(convo["content"])
        print(f"Similarity score: {convo['similarity_score']}")
        print("-" * 50)

if __name__ == "__main__":
    print("\n\n\n")
    test_usage()
