from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"


def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    #embeddings
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    #database
    vector_store = Chroma.from_documents(documents=splits,embedding=embedding)


    return vector_store



def create_rag_chain(vector_store):
    llm = OllamaLLM(model=LLM_MODEL)

    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Your goal is to provide a detailed and comprehensive answer.
    Extract all relevant information from the context to formulate your response.
    Think step by step and structure your answer logically.
    If the context does not contain the answer to the question, state that the information is not available in the provided document. Do not attempt to make up information.

    <context>
    {context}
    </context>

    Question: {question}
    """)

    # Create retriever
    retriever = vector_store.as_retriever()

    # Create a simple RAG chain using LCEL (LangChain Expression Language)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create a proper RAG chain class that matches the expected interface
    class RAGChain:
        def invoke(self, inputs):
            # Extract question from inputs dict
            question = inputs.get("input", "")
            
            # Get relevant documents
            docs = retriever.invoke(question)
            context = format_docs(docs)
            
            # Format the prompt
            formatted_prompt = prompt.format(context=context, question=question)
            
            # Get response from LLM
            response = llm.invoke(formatted_prompt)
            
            # Return in expected format
            return {
                "answer": response,
                "context": docs
            }

    return RAGChain()

