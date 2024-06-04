from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough

from llm_chain import LLM_Chain
from document_chain import DocumentProcessor

import time
import argparse

import warnings
warnings.filterwarnings("ignore")

class AskMyPDFAgent:
    """
    A class to create an agent that answers questions using a combination of document retrieval and language model processing.

    Attributes:
        llm_chain (LLM_Chain): The language model chain for generating answers.
        document_processor (DocumentProcessor): The document processor for retrieving relevant context from documents.

    Methods:
        answer_question(query, useRAG=True):
            Answers a question using either a Retrieval-Augmented Generation (RAG) approach or a direct LLM invocation.
    """
    
    def __init__(self, llmchain, document_processor):
        """
        Initializes the AskMyPDFAgent with a language model chain and a document processor.

        Args:
            llmchain (LLM_Chain): The language model chain for generating answers.
            document_processor (DocumentProcessor): The document processor for retrieving relevant context from documents.
        """
        self.llm_chain = llmchain
        self.document_processor = document_processor

    def answer_question(self, query, useRAG=True):
        """
        Answers a question using either a Retrieval-Augmented Generation (RAG) approach or a direct LLM invocation.

        Args:
            query (str): The question to be answered.
            useRAG (bool): Flag to determine if RAG should be used. Default is True.

        Returns:
            dict: The generated answer to the query.
        """
        if useRAG:
            retriever = self.document_processor.db.as_retriever()
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()} | self.llm_chain.chain
            )
            return rag_chain.invoke(query)
        else:
            return self.llm_chain.chain.invoke({"context": "", "question": query})


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--document", type=str, default="")
    
    args = arg_parser.parse_args()

    # check if the document path is provided
    if args.document == "":
        print("Please provide a valid path for a document.")
        exit()
    
    document_processor = DocumentProcessor(args.document)
    llm_chain = LLM_Chain(llm_model_path='mistralai/Mistral-7B-Instruct-v0.3')

    agent = AskMyPDFAgent(llm_chain, document_processor)
    
    while True:
        query = input("\n\nEnter your question: ")

        if query.lower() == "exit":
            break

        print("Processing...")
        t1 = time.time()
        answer = agent.answer_question(query, useRAG=True)
        t2 = time.time()
        
        answer = answer["text"]

        # Print time to answer in milliseconds
        print("Time to answer: ", (t2-t1), "secs")

        # Define the substring to search for
        substring = '[/INST]\n'

        # Find the position of the substring
        position = answer.find(substring)

        # Extract everything after the substring
        if position != -1:
            filtered_answer = answer[position + len(substring):]
            filtered_answer = filtered_answer.strip()
        else:
            filtered_answer = "The substring was not found."
        
        print("Filtered Answer: \n", filtered_answer)

        
