# Load packages
import os
from time import sleep

# Declare functions
def SendDeepResearchQuery(query_text,
                          openai_api_key=None,
                          plan_model_name='gpt-5-mini',
                          research_model_name='o4-mini-deep-research',
                          model_name='o4-mini-deep-research',
                          verbose=True,
                          research_tools=[
                            {"type": "web_search_preview"},
                        ],
                          skip_clarification=False):
    """
    Send a deep research query to OpenAI's Deep Research API with clarifying questions.
    
    This function uses a three-phase approach:
    1. First, asks clarifying questions to gather detailed information from the user
    2. Then, pauses to wait for user input. If no input is provided, continues with AI-generated plan
    3. Finally, creates a comprehensive research plan and sends it to OpenAI's Deep Research API
    
    Parameters:
    -----------
    query_text : str
        The initial research query to investigate (e.g., "Research surfboards for me")
    openai_api_key : str, optional
        OpenAI API key. If not provided, will raise an error.
    model_name : str, default='o4-mini-deep-research'
        The model to use for the research query.
    max_tokens : int, default=4000
        Maximum number of tokens in the response.
    temperature : float, default=0.0
        Controls randomness in the response (0.0 = deterministic).
    verbose : bool, default=True
        Whether to print status messages.
    skip_clarification : bool, default=False
        If True, skips the clarification phase and goes directly to research.
    
    Returns:
    --------
    dict
        A dictionary containing the research results with the following structure:
        {
            'query': str,                    # The original query
            'clarifying_questions': str,     # Questions asked to the user
            'user_additional_info': str,     # Additional information provided by user (if any)
            'enhanced_query': str,           # The query enhanced with user input
            'research_plan': str,            # The final research plan created
            'research_summary': str,         # Comprehensive research summary
            'key_findings': list,            # List of key findings
            'sources': list,                 # List of sources used
            'metadata': dict                 # Additional metadata about the research
        }
    
    Raises:
    -------
    ValueError
        If no API key is provided or if query_text is empty.
    Exception
        If there's an error with the API call.
    """
    # Lazy load uncommon packages
    import openai
    import json
    from typing import Dict, Any
    
    # Validate inputs
    if openai_api_key is None:
        # Get the API key from the environment variable
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        except:
            raise ValueError("No API key provided. If you need an OpenAI API key, visit https://platform.openai.com/")
    
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty.")
    
    # Clean the query text
    query_text = query_text.strip()
    
    if verbose:
        print(f"Starting deep research process for: {query_text[:100]}...")
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        clarifying_questions = ""
        research_plan = ""
        
        # Phase 1: Ask clarifying questions (unless skipped)
        if not skip_clarification:
            if verbose:
                print("Phase 1: Generating clarifying questions...")
            
            clarification_instructions = """
You are talking to a user who is asking for a research task to be conducted. Your job is to gather more information from the user to successfully complete the task.

GUIDELINES:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Use bullet points or numbered lists if appropriate for clarity.
- Don't ask for unnecessary information, or information that the user has already provided.

IMPORTANT: Do NOT conduct any research yourself, just gather information that will be given to a researcher to conduct the research task.
"""
            
            clarification_response = client.responses.create(
                model=plan_model_name,
                instructions=clarification_instructions,
                input=query_text,
                max_output_tokens=1000,
            )
            
            # Get the clarifying questions from the response
            clarifying_questions = clarification_response.output_text
            # return clarification_response
            
            if verbose:
                print("Clarifying questions generated. Please provide additional details based on these questions.")
                print(f"Questions: {clarifying_questions}")
            
            # Wait for user input after showing clarifying questions
            print("\n" + "="*60)
            print("Please provide additional details based on the questions above.")
            print("Press Enter without typing anything to continue with AI-generated research plan.")
            print("="*60)
            
            try:
                user_additional_info = input("Your response (or press Enter to continue): ").strip()
            except (EOFError, KeyboardInterrupt):
                # Handle cases where input is not available (e.g., in automated environments)
                user_additional_info = ""
                if verbose:
                    print("No user input received, continuing with AI-generated research plan...")
            
            # If user provided additional information, incorporate it into the query
            if user_additional_info:
                if verbose:
                    print("User provided additional information. Incorporating into research plan...")
                # Combine original query with user's additional information
                enhanced_query = f"{query_text}\n\nAdditional user information: {user_additional_info}"
            else:
                if verbose:
                    print("No additional information provided. Continuing with original query...")
                enhanced_query = query_text
        
        # Phase 2: Create research plan
        if verbose:
            print("Phase 2: Creating comprehensive research plan...")
        
        research_plan_instructions = """
You will be given a research task by a user. Your job is to produce a set of
instructions for a researcher that will complete the task. Do NOT complete the
task yourself, just provide instructions on how to complete it.

GUIDELINES:
1. **Maximize Specificity and Detail**
- Include all known user preferences and explicitly list key attributes or
  dimensions to consider.
- It is of utmost importance that all details from the user are included in
  the instructions.

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
- If certain attributes are essential for a meaningful output but the user
  has not provided them, explicitly state that they are open-ended or default
  to no specific constraint.

3. **Avoid Unwarranted Assumptions**
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat
  it as flexible or accept all possible options.

4. **Use the First Person**
- Phrase the request from the perspective of the user.

5. **Tables**
- If you determine that including a table will help illustrate, organize, or
  enhance the information in the research output, you must explicitly request
  that the researcher provide them.

Examples:
- Product Comparison (Consumer): When comparing different smartphone models,
  request a table listing each model's features, price, and consumer ratings
  side-by-side.
- Project Tracking (Work): When outlining project deliverables, create a table
  showing tasks, deadlines, responsible team members, and status updates.
- Budget Planning (Consumer): When creating a personal or household budget,
  request a table detailing income sources, monthly expenses, and savings goals.
- Competitor Analysis (Work): When evaluating competitor products, request a
  table with key metrics, such as market share, pricing, and main differentiators.

6. **Headers and Formatting**
- You should include the expected output format in the prompt.
- If the user is asking for content that would be best returned in a
  structured format (e.g. a report, plan, etc.), ask the researcher to format
  as a report with the appropriate headers and formatting that ensures clarity
  and structure.

7. **Language**
- If the user input is in a language other than English, tell the researcher
  to respond in this language, unless the user query explicitly asks for the
  response in a different language.

8. **Sources**
- If specific sources should be prioritized, specify them in the prompt.
- For product and travel research, prefer linking directly to official or
  primary websites (e.g., official brand sites, manufacturer pages, or
  reputable e-commerce platforms like Amazon for user reviews) rather than
  aggregator sites or SEO-heavy blogs.
- For academic or scientific queries, prefer linking directly to the original
  paper or official journal publication rather than survey papers or secondary
  summaries.
- If the query is in a specific language, prioritize sources published in that
  language.
"""
        
        # Use the enhanced query (with user input) or original query for research plan creation
        query_for_plan = enhanced_query if 'enhanced_query' in locals() else query_text
        plan_response = client.responses.create(
            model=plan_model_name,
            instructions=research_plan_instructions,
            input=query_for_plan,
            max_output_tokens=2000,
        )
        
        # Get the research plan from the response
        research_plan = plan_response.output_text
        
        if verbose:
            print("Research plan created successfully. Here it is:")
            print(research_plan)
        
        # Phase 3: Execute the research using Deep Research API
        if verbose:
            print("Phase 3: Executing research using Deep Research API...")
        
        # Create the research query using the Deep Research API
        research_client = openai.OpenAI(api_key=openai_api_key, timeout=3600)
        response = research_client.responses.create(
            input=research_plan,
            model=research_model_name,
            # background=True,
            # stream=True,
            tools=research_tools,
        )
        
        # Stream the response
        if verbose:
            # while response.status in {"queued", "in_progress"}:
            #     print(f"Current status: {response.status}")
            #     sleep(2)
            #     resp = research_client.responses.retrieve(resp.id)
            #     print(f"Final status: {resp.status}\nOutput:\n{resp.output_text}")
            print("Research query completed successfully.")
            
        # Show the report in Markdown format
        from IPython.display import Markdown
        Markdown(response.output_text)
        
        # Extract the research results
        research_data = response.model_dump()
        
        # Structure the response in a user-friendly format
        import datetime
        result = {
            'query': query_text,
            'clarifying_questions': clarifying_questions,
            'user_additional_info': user_additional_info if 'user_additional_info' in locals() else "",
            'enhanced_query': enhanced_query if 'enhanced_query' in locals() else query_text,
            'research_plan': research_plan,
            'research_results': response.output_text,
            'metadata': {
                'model_used': model_name,
                'tokens_used': research_data.get('usage', {}).get('total_tokens', 0),
                'research_id': research_data.get('id', ''),
                'created_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': research_data.get('status', ''),
            }
        }
        
        # Extract sources
        sources_dict = {}
        annotations = response.output[-1].content[0].annotations
        for i, citation in enumerate(annotations):
            sources_dict[i] = {
                'title': citation.title,
                'url': citation.url,
                'location': f"chars {citation.start_index}–{citation.end_index}"
            }
        result['sources'] = sources_dict
        
        # Return the results
        return result
        
    except openai.APIError as e:
        error_msg = f"OpenAI API error: {str(e)}"
        if verbose:
            print(f"Error: {error_msg}")
        raise Exception(error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error during deep research query: {str(e)}"
        if verbose:
            print(f"Error: {error_msg}")
        raise Exception(error_msg)
