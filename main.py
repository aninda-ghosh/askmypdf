from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough

from llm_chain import LLM_Chain  # Custom LLM Chain class for managing LLMs
from document_chain import DocumentProcessor  # Custom Document Processor class for handling documents

import time
import argparse

import warnings
warnings.filterwarnings("ignore")

class AskMyPDFAgent:
    """
    Agent class to interact with an LLM and process documents to answer questions.
    """
    def __init__(self, llmchain, document_processor):
        """
        Initializes the agent with a language model chain and a document processor.
        
        :param llmchain: An instance of LLM_Chain to manage the language model.
        :param document_processor: An instance of DocumentProcessor to handle document processing.
        """
        self.llm_chain = llmchain
        self.document_processor = document_processor

    def answer_question(self, query, useRAG=True):
        """
        Answers a question using the language model and optionally the document retriever.
        
        :param query: The question to be answered.
        :param useRAG: Boolean to indicate whether to use Retrieval-Augmented Generation (RAG).
        :return: The answer to the query.
        """
        if useRAG:
            # Use the document retriever for context
            retriever = self.document_processor.db.as_retriever()
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()} | self.llm_chain.chain
            )
            return rag_chain.invoke(query)
        else:
            # Directly use the LLM without retrieval
            return self.llm_chain.chain.invoke({"context": "", "question": query})


if __name__ == "__main__":
    # Argument parser for command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-m", "--model", type=str, default="mistral")
    arg_parser.add_argument("-d", "--document", type=str, default="")
    
    args = arg_parser.parse_args()

    # Check if the document path is provided
    if args.document == "":
        print("Please provide a valid path for a document.")
        exit()
    
    # Initialize LLM and document processor based on the model choice
    if args.model == "mistral":
        llm_chain = LLM_Chain(llm_model_path='mistralai/Mistral-7B-Instruct-v0.3')
        document_processor = DocumentProcessor(args.document, chunk_size=1000, chunk_overlap=200)
    elif args.model == "gemma":
        llm_chain = LLM_Chain(llm_model_path='google/gemma-1.1-2b-it', enable_quantization=False)
        document_processor = DocumentProcessor(args.document, chunk_size=2000, chunk_overlap=500)

    # Create an instance of the agent
    agent = AskMyPDFAgent(llm_chain, document_processor)
    
    while True:
        # Prompt user for a question
        query = input("\n\nEnter your question: ")

        # Exit the loop if user types 'exit'
        if query.lower() == "exit":
            break

        print("Processing...")
        t1 = time.time()  # Start time
        answer = agent.answer_question(query, useRAG=True)  # Get the answer
        t2 = time.time()  # End time
        
        answer = answer["text"]  # Extract the answer text

        # Print time taken to answer in seconds
        print("Time to answer: ", (t2-t1), "secs")

        # Define the substring to search for in the answer
        substring = '[/INST]\n'

        # Find the position of the substring in the answer
        position = answer.find(substring)

        # Extract everything after the substring
        if position != -1:
            filtered_answer = answer[position + len(substring):].strip()
        else:
            filtered_answer = "The substring was not found."
        
        # Print the filtered answer
        print("Filtered Answer: \n", filtered_answer)
