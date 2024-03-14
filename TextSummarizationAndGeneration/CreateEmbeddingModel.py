# Load packages

# Declare function
def CreateEmbeddingModel(model_name):
    """Create a sentence embedding model using the sentence_transformers package.
    
    Args:
        model_name: str, the name of the model to use.
    """
    # Lazy load uncommon packages
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Check if cuda is available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = 'cuda'
    else:
        print("CUDA is not available. Using CPU.")
        device = 'cpu'
    
    # Load the model
    model = SentenceTransformer(
        'all-MiniLM-L6-v2', 
        device=device
    )

    # Return the model
    return model

