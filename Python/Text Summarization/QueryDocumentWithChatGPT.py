from IPython.display import display, Markdown
import json
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, JSONLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS, DocArrayInMemorySearch
import openai
import tiktoken

# Declare the function
def QueryDocumentWithChatGPT(document_filepath,
                             my_query,
                             openai_api_key,
                             query_method="stuff",
                             temperature=0.0,
                             verbose=True,
                             json_field=None,
                             csv_source_column=None,
                             return_vectorstore=False):
    
    # Ensure that the query method is valid
    if query_method not in ["stuff", "map_reduce", "refine"]:
        error_message = f"""
        query_method must be one of: stuff, map_reduce, refine.
        - stuff: makes a single query to the model and it has access to all data as context
        - map_reduce: breaks documents into independent chunks and calls, then a final call is made to reduce all responses to a final answer. Best for summarizing really long documents.
        - refine: break documents into chunks, but each response is based on the previous.
        """
        raise ValueError()

    # If the document is a PDF, use the PyPDFLoader
    if document_filepath.endswith(".pdf"):
        # Load PDF using pypdf into array of documents, where each document contains the page content and metadata with page number.
        loader = PyPDFLoader(document_filepath)
        data = loader.load_and_split()
    # If the document is a JSON, use the JSONLoader
    elif document_filepath.endswith(".json") or document_filepath.endswith(".jsonl"):
        # See if JSON is in JSONL format
        if document_filepath.endswith(".jsonl"):
            json_lines = True
        else:
            json_lines = False
        # The JSONLoader uses a specified jq schema to parse the JSON files. It uses the jq python package. 
        loader = JSONLoader(
            file_path=document_filepath,
            jq_schema='.content',
            json_lines=json_lines
        )
        data = loader.load_and_split()
    # If the document is a CSV, use the CSVLoader
    elif document_filepath.endswith(".csv"):
        loader = CSVLoader(
            file_path=document_filepath,
            source_column=csv_source_column
        )
        data = loader.load_and_split()

    # Embed the documents
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = DocArrayInMemorySearch.from_documents(
        data, 
        embeddings
    )
    
    # Create index of documents
    if return_vectorstore:
        index = VectorstoreIndexCreator(
            vectorstore_cls=DocArrayInMemorySearch,
            embeddings=embeddings
        ).from_loaders([loader])

    # Set the retrieval QA model
    retriever = db.as_retriever()

    # Set the model name and temperature
    chatgpt_model = ChatOpenAI(
        temperature=0.0,
        openai_api_key=openai_api_key
    )

    # Create the model and run the query
    query_retrieval = RetrievalQA.from_chain_type(
        llm=chatgpt_model, 
        chain_type=query_method, 
        retriever=retriever,
        verbose=verbose
    )
    response = query_retrieval.run(my_query)

    # Return the response
    if return_vectorstore:
        return(response, index)
    else:
        return(response)


# # Test the function
# response = QueryDocumentWithChatGPT(
#     document_filepath="C:/Users/oneno/OneDrive/Creations/Star Sense/StarSense/PDF/QRS/Technical Specifications/2022_QRS_Measure_Technical_Specifications_508.pdf",
#     my_query="""
#     What is the list of measures? 
#     Include the measure name, the measure steward, and the NQF ID. 
#     If a measure is denoted with a caret (^), mark it as "1" in the MAY_BE_REMOVED column.
#     If a measure is denoted with an asterisk (*), mark it as "1" in the NOT_SCORED column.
#     Structure the reponse as a CSV with the columns: MEASURE_NAME, MEASURE_STEWARD, NQF_ID, MAY_BE_REMOVED, and NOT_SCORED.
#     Separate each column with a semi-colon (;).
#     """,
#     openai_api_key=open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# )
# print(response)
