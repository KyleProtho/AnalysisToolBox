# Load packages

# Declare function
def ExtractTranscriptFromYouTube(youtube_url,
                                 add_video_info=True,
                                 print_transcript=False,
                                 # LLM parameters
                                 openai_api_key=None):
    # Lazy load the YoutubeLoader
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_community.document_loaders.youtube import TranscriptFormat
    
    # Load the YoutubeLoader
    loader = YoutubeLoader.from_youtube_url(
        youtube_url, 
        add_video_info=add_video_info,
        transcript_format=TranscriptFormat,
        # chunk_size_seconds=chunk_size_in_seconds,
    )
    
    # Extract the transcript
    transcript = "\n\n".join(map(repr, loader.load()))
    
    # Print the transcript if requested
    if print_transcript:
        print(transcript)
    
    # Return the transcript
    return transcript


# Example usage
# Import OpenAI key
import os
openai_api_key = os.environ.get("OPENAI_API_KEY")

youtube_url = "https://www.youtube.com/watch?v=WONRS7BLh4g"
transcript = ExtractTranscriptFromYouTube(
    youtube_url, 
    add_video_info=True,
    print_transcript=True,
    openai_api_key=openai_api_key
)