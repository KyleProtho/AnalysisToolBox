# Load packages
from dotenv import load_dotenv
from IPython.display import display, HTML, Markdown
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Declare function
def SummarizeTextWithChatGPT(text_to_summarize,
                             openai_api_key=None,
                             # LLM parameters
                             summarization_objective="Summarize the main points and key arguments of the given text.",
                             query_method="map_reduce",
                             temperature=0.0,
                             chat_model_name="gpt-3.5-turbo",
                             verbose=False,
                             # Text parsing parameters
                             splitter_chunk_size=None,
                             splitter_chunk_overlap=None,
                             splitter_separators=[".  ", "\n", "\n\n"]):
    # If OpenAI API key is not provided, then try to load from .env file
    if openai_api_key is None:
        load_dotenv()
        try:
            openai_api_key = os.environ['OPENAI_API_KEY']
        except:
            raise ValueError("No API key provided and no .env file found. If you need a OpenAI API key, visit https://platform.openai.com/")
    
    # If chunk size is not provided, then set defaults
    if splitter_chunk_size is None:
        if "gpt-3.5-turbo-16k" in chat_model_name:
            splitter_chunk_size = 16000 - 4000
        elif "gpt-4-32k" in chat_model_name:
            splitter_chunk_size = 32000 - 4000
        elif "gpt-4" in chat_model_name:
            splitter_chunk_size = 8000 - 3000
        else:
            splitter_chunk_size = 4000 - 2500
    
    # If splitter chunk overlap is not provided, then set defaults
    if splitter_chunk_overlap is None:
        splitter_chunk_overlap = splitter_chunk_size // 10
        
    # If no API key is provided, try to load from .env file
    if openai_api_key is None:
        load_dotenv()
        try:
            openai_api_key = os.environ['OPENAI_API_KEY']
        except:
            raise ValueError("No API key provided and no .env file found. If you need a OpenAI API key, visit https://platform.openai.com/")
    
    # Ensure that the query method is valid
    if query_method not in ["stuff", "map_reduce", "refine"]:
        error_message = f"""
        query_method must be one of: stuff, map_reduce, or refine.
        - stuff: makes a single query to the model and it has access to all data as context
        - map_reduce: breaks documents into independent chunks and calls, then a final call is made to reduce all responses to a final answer. Best for summarizing really long documents.
        - refine: break documents into chunks, but each response is based on the previous.
        """
        raise ValueError(error_message)
    
    # Setup the splitter
    splitter = RecursiveCharacterTextSplitter(
        separators=splitter_separators,
        chunk_size=splitter_chunk_size,
        chunk_overlap=splitter_chunk_overlap
    )
    
    # Ensure that text_to_summarize is a string or filepath to .txt, .doc, .docx, or .pdf file
    if not isinstance(text_to_summarize, str):
        raise ValueError("text_to_summarize must be a string or a filepath to a .txt, .doc, .docx, or .pdf file.")
    
    # Load in the text to summarize
    # If text_to_summarize is a filepath, then make sure it ends with .txt, .doc, .docx, or .pdf file
    if text_to_summarize.endswith(".txt"):
        # If text_to_summarize is a .txt file, then read the file
        loader = TextLoader(document_filepath_or_url)
        text = loader.load()
        docs = splitter.create_documents([text])
    # If text to summarize is a .doc or .docx file, then use the Docx2txtLoader
    elif text_to_summarize.endswith(".doc") or text_to_summarize.endswith(".docx"):
        loader = Docx2txtLoader(text_to_summarize)
        text = loader.load()
        docs = splitter.create_documents([text])
    # If text_to_summarize is a .pdf file, then load it
    elif text_to_summarize.endswith(".pdf"):
        loader = PyPDFLoader(text_to_summarize)
        text = loader.load()
        docs = splitter.create_documents([text])
    else:
        # If text_to_summarize is a string, then it is the text to summarize
        text = text_to_summarize
        docs = splitter.create_documents([text])
    
    # Set up the LLM
    llm = ChatOpenAI(api_key=openai_api_key,
                     model_name=chat_model_name)
    
    # Set up the prompt template
    map_prompt = """
    {summarization_objective}
    
    TEXT:
    "{text}"
    
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt,
        input_variables=["summarization_objective", "text"],
    )
    
    # Create summarization chain
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type=query_method,
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=verbose
    )
    
    # Generate the summary
    summary = summary_chain.run(
        input_documents=docs,
        summarization_objective=summarization_objective
    )
    
    # Return the summary
    return summary


# # Test the function
# long_text = """
# Multi-agent Conversation Framework | AutoGen


# Getting StartedInstallationUse CasesMulti-agent Conversation FrameworkEnhanced InferenceExamplesContributingResearchOn this pageMulti-agent Conversation FrameworkAutoGen offers a unified multi-agent conversation framework as a high-level abstraction of using foundation models. It features capable, customizable and conversable agents which integrate LLM, tool and human via automated agent chat.
# By automating chat among multiple capable agents, one can easily make them collectively perform tasks autonomously or with human feedback, including tasks that require using tools via code.This framework simplifies the orchestration, automation and optimization of a complex LLM workflow. It maximizes the performance of LLM models and overcome their weaknesses. It enables building next-gen LLM applications based on multi-agent conversations with minimal effort.Agents​AutoGen abstracts and implements conversable agents
# designed to solve tasks through inter-agent conversations. Specifically, the agents in AutoGen have the following notable features:Conversable: Agents in AutoGen are conversable, which means that any agent can send
# and receive messages from other agents to initiate or continue a conversationCustomizable: Agents in AutoGen can be customized to integrate LLMs, humans, tools, or a combination of them.The figure below shows the built-in agents in AutoGen.
# We have designed a generic ConversableAgent class for Agents that are capable of conversing with each other through the exchange of messages to jointly finish a task. An agent can communicate with other agents and perform actions. Different agents can differ in what actions they perform after receiving messages. Two representative subclasses are AssistantAgent and UserProxyAgent.The AssistantAgent is designed to act as an AI assistant, using LLMs by default but not requiring human input or code execution. It could write Python code (in a Python coding block) for a user to execute when a message (typically a description of a task that needs to be solved) is received. Under the hood, the Python code is written by LLM (e.g., GPT-4). It can also receive the execution results and suggest corrections or bug fixes. Its behavior can be altered by passing a new system message. The LLM inference configuration can be configured via llm_config.The UserProxyAgent is conceptually a proxy agent for humans, soliciting human input as the agent's reply at each interaction turn by default and also having the capability to execute code and call functions. The UserProxyAgent triggers code execution automatically when it detects an executable code block in the received message and no human user input is provided. Code execution can be disabled by setting the code_execution_config parameter to False. LLM-based response is disabled by default. It can be enabled by setting llm_config to a dict corresponding to the inference configuration. When llm_config is set as a dictionary, UserProxyAgent can generate replies using an LLM when code execution is not performed.The auto-reply capability of ConversableAgent allows for more autonomous multi-agent communication while retaining the possibility of human intervention.
# One can also easily extend it by registering reply functions with the register_reply() method.In the following code, we create an AssistantAgent named "assistant" to serve as the assistant and a UserProxyAgent named "user_proxy" to serve as a proxy for the human user. We will later employ these two agents to solve a task.from autogen import AssistantAgent, UserProxyAgent# create an AssistantAgent instance named "assistant"assistant = AssistantAgent(name="assistant")# create a UserProxyAgent instance named "user_proxy"user_proxy = UserProxyAgent(name="user_proxy")CopyMulti-agent Conversations​A Basic Two-Agent Conversation Example​Once the participating agents are constructed properly, one can start a multi-agent conversation session by an initialization step as shown in the following code:# the assistant receives a message from the user, which contains the task descriptionuser_proxy.initiate_chat(    assistant,    message="What date is today? Which big tech stock has the largest year-to-date gain this year? How much is the gain?",)CopyAfter the initialization step, the conversation could proceed automatically. Find a visual illustration of how the user_proxy and assistant collaboratively solve the above task autonmously below:
# The assistant receives a message from the user_proxy, which contains the task description.The assistant then tries to write Python code to solve the task and sends the response to the user_proxy.Once the user_proxy receives a response from the assistant, it tries to reply by either soliciting human input or preparing an automatically generated reply. If no human input is provided, the user_proxy executes the code and uses the result as the auto-reply.The assistant then generates a further response for the user_proxy. The user_proxy can then decide whether to terminate the conversation. If not, steps 3 and 4 are repeated.Supporting Diverse Conversation Patterns​Conversations with different levels of autonomy, and human-involvement patterns​On the one hand, one can achieve fully autonomous conversations after an initialization step. On the other hand, AutoGen can be used to implement human-in-the-loop problem-solving by configuring human involvement levels and patterns (e.g., setting the human_input_mode to ALWAYS), as human involvement is expected and/or desired in many applications.Static and dynamic conversations​By adopting the conversation-driven control with both programming language and natural language, AutoGen inherently allows dynamic conversation. Dynamic conversation allows the agent topology to change depending on the actual flow of conversation under different input problem instances, while the flow of a static conversation always follows a pre-defined topology. The dynamic conversation pattern is useful in complex applications where the patterns of interaction cannot be predetermined in advance. AutoGen provides two general approaches to achieving dynamic conversation:Registered auto-reply. With the pluggable auto-reply function, one can choose to invoke conversations with other agents depending on the content of the current message and context. A working system demonstrating this type of dynamic conversation can be found in this code example, demonstrating a dynamic group chat. In the system, we register an auto-reply function in the group chat manager, which lets LLM decide who the next speaker will be in a group chat setting.LLM-based function call. In this approach, LLM decides whether or not to call a particular function depending on the conversation status in each inference call.
# By messaging additional agents in the called functions, the LLM can drive dynamic multi-agent conversation. A working system showcasing this type of dynamic conversation can be found in the multi-user math problem solving scenario, where a student assistant would automatically resort to an expert using function calls.Diverse Applications Implemented with AutoGen​The figure below shows six examples of applications built using AutoGen.
# Automated Task Solving with Code Generation, Execution & DebuggingAuto Code Generation, Execution, Debugging and Human FeedbackSolve Tasks Requiring Web InfoUse Provided Tools as FunctionsAutomated Task Solving with Coding & Planning AgentsAutomated Task Solving with GPT-4 + Multiple Human UsersAutomated Chess Game Playing & Chitchatting by GPT-4 AgentsAutomated Task Solving by Group Chat (with 3 group member agents and 1 manager agent)Automated Data Visualization by Group Chat (with 3 group member agents and 1 manager agent)Automated Complex Task Solving by Group Chat (with 6 group member agents and 1 manager agent)Automated Continual Learning from New DataTeach Agents New Skills & Reuse via Automated ChatAutomated Code Generation and Question Answering with Retrieval Augemented AgentsFor Further Reading​Interested in the research that leads to this package? Please check the following papers.AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework. Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang and Chi Wang. ArXiv 2023.An Empirical Study on Challenging Math Problem Solving with GPT-4. Yiran Wu, Feiran Jia, Shaokun Zhang, Hangyu Li, Erkang Zhu, Yue Wang, Yin Tat Lee, Richard Peng, Qingyun Wu, Chi Wang. ArXiv preprint arXiv:2306.01337 (2023).Edit this pagePrevious« InstallationNextEnhanced Inference »CommunityDiscordTwitterCopyright © 2023 AutoGen Authors.

# """
# summary_output = SummarizeTextWithChatGPT(
#     text_to_summarize=long_text,
#     openai_api_key=open("C:/Users/oneno/OneDrive/Desktop/OpenAI key.txt", "r").read()
# )
