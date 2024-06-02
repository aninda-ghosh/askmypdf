from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class DocumentProcessor:
    """
    A class to process documents and enable semantic search using FAISS and HuggingFace embeddings.

    Attributes:
        document (list): Loaded document content.
        sentence_processor (HuggingFaceEmbeddings): Embedding model for generating sentence embeddings.
        db (FAISS): FAISS index for storing and retrieving document chunks.

    Methods:
        __init__(document_path):
            Initializes the DocumentProcessor with a specified document path, processes the document into chunks, 
            and loads the chunks into the FAISS index.
        
        search(query):
            Searches the FAISS index for the most similar document chunks to the query.

    """
    
    def __init__(self, document_path):
        """
        Initializes the DocumentProcessor with a specified document path.

        Args:
            document_path (str): Path to the document to be processed.
        """
        loader = PyPDFLoader(document_path)
        self.document = loader.load()
        self.sentence_processor = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        # Split documents into text and embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        chunked_documents = text_splitter.split_documents(self.document)
        
        # Load chunked documents into the FAISS index
        self.db = FAISS.from_documents(chunked_documents, self.sentence_processor)

    def search(self, query):
        """
        Searches the FAISS index for the most similar document chunks to the query.

        Args:
            query (str): The query string to search for in the document chunks.

        Returns:
            list: A list of the most similar document chunks to the query.
        """
        retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 4}, 
            score_threshold=0.9
        )
        return retriever.search(query)
