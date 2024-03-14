# Load packages
import os
import time
import sys

# Declare function
def CreatePineconeIndex(pinecone_api_key,
                        pinecone_index_name,
                        openai_api_key,
                        sentence_transformer_model,
                        # Pinecone index arguments
                        dimension_count=384,
                        # Pinecode pod-based spec arguments
                        environment=None,
                        # Pinecone serverless spec arguments
                        cloud_provider='aws',
                        cloud_server_region='us-west-2',
                        metric='cosine'):
    # Lazy load uncommon packages
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone, ServerlessSpec
    from tqdm.auto import tqdm
    
    # Setup Pinecone
    pinecone = Pinecone(api_key=PINECONE_API_KEY)
    
    # See if index already exists
    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in existing_indexes:
        # Get the dimension count
        dimension_count = model.get_sentence_embedding_dimension()
        
        # Create a Pinecone index name with the OpenAI API key
        INDEX_NAME = f'{pinecone_index_name}-{openai_api_key[-36:].lower().replace("_", "-")}'
    
        # if does not exist, create index
        pinecone.create_index(
            name=INDEX_NAME, 
            dimension=dimension_count,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud_provider, 
                region=cloud_server_region
            )
        )
        
        # wait for index to be initialized
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)
            
        # Create a Pinecone index
        index = pinecone.Index(INDEX_NAME)
    
    # Print and return the Pinecone index name
    print(f'Pinecone index created: {INDEX_NAME}')
    return index
