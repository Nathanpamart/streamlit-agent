from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI

import streamlit as st


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

st.set_page_config(page_title="UKIO Chatbot Demo", page_icon="📖")
st.title("📖 UKIO Chatbot Demo")

"""
A basic example of a chatbot to help customers learn more about UKIO's apartments through a natural language interface.
Currently, the bot is equipped with the ability to get information from UKIO's website, and from web search.
It doesn't have any additional information.
"""

# Set up session state

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

#view_messages = st.expander("View the message contents in session state")

# Get an OpenAI API Key before continuing
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets.OPENAI_API_KEY
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

# Set up the LangChain, passing in Message History

from langchain_openai import ChatOpenAI

import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

def scrape_with_playwright(urls):
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    html2text = Html2TextTransformer()
    html2text.ignore_links = False
    docs_transformed = html2text.transform_documents(docs)

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)
    
    return splits

# Import things that are needed generically
from typing import Optional, Type
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

class UkioSearchInput(BaseModel):
    city: str = Field(description="the name of city to search for apartments in. It can only accept one city at a time. The following cities are available: Madrid, Barcelona, Paris, Lisbon, Berlin. Any other city is not available and will return an error.")
    bedrooms: Optional[int] = Field(description="the minimum number of bedrooms (can be 0 if I ask for a studio")
    check_in: Optional[str] = Field(description="the check in date, format yyyy-mm-dd")
    check_out: Optional[str] = Field(description="the check out date, format yyyy-mm-dd")
    rent_from: Optional[int] = Field(description="the minimum price per month in euros")
    rent_to: Optional[int] = Field(description="the maximum price per month in euros")

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class UkioSearchInputRun(BaseTool):
    """Tool that searche the Ukio website for apartments meeting the criteria."""

    name: str = "ukio_search"
    description: str = (
        "Tool that searche the Ukio website for apartments meeting the criteria."
    )
    args_schema: Type[BaseModel] = UkioSearchInput

    def _run(
        self,
        city: str, 
        bedrooms: Optional[int] = None, 
        check_in: Optional[str] = None, 
        check_out: Optional[str] = None, 
        rent_from : Optional[int] = None, 
        rent_to: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        url = "https://ukio.com/apartments/"+city.lower()+"?"
        if bedrooms:
            url+="&bedrooms="+str(bedrooms)
        if rent_from:
            url+="&rent_from="+str(rent_from)
        if rent_to:
            url+="&rent_to="+str(rent_to)
        if check_in:
            url+="&check_in="+check_in
        if check_out:
            url+="&check_out="+check_out
        print(f"About to scrape {url} from the tool...")
        response = scrape_with_playwright([url])
        
        return response

class UkioBrowseApartmentInput(BaseModel):
    url: str = Field(description="the url of the apartment for which you are looking for more details. it can usually be found in the message history")

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class BrowseApartmentUrl(BaseTool):
    """Tool that browses to the url of an apartment to find more details about it. Use it when the user asks about information specific to one apartment"""

    name: str = "browse_apartment_details"
    description: str = (
        "Tool that browses to the url of an apartment to find more details about it. Use it when the user asks about information specific to one apartment."
    )
    args_schema: Type[BaseModel] = UkioBrowseApartmentInput

    def _run(
        self,
        url: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:

        url = url
        print(f"About to scrape {url} from the tool...")
        response = scrape_with_playwright([url])
        
        return response


from langchain_community.tools import DuckDuckGoSearchRun

search_ukio_apartments_tool = UkioSearchInputRun()
get_apartment_details_tool = BrowseApartmentUrl()
web_search_tool = DuckDuckGoSearchRun()

tools = [search_ukio_apartments_tool, get_apartment_details_tool, web_search_tool]

from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")

from langchain.agents import create_openai_functions_agent
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", streaming=True)
agent = create_openai_functions_agent(llm, tools, prompt)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():

    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "<foo>"}}
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm.callbacks = [stream_handler]
        response = chain_with_history.invoke({"input": prompt}, config)

# Draw the messages at the end, so newly generated ones show up immediately
#with view_messages:
#    """
#    Message History initialized with:
#    ```python
#    msgs = StreamlitChatMessageHistory(key="langchain_messages")
#    ```
#
#    Contents of `st.session_state.langchain_messages`:
#    """
#    view_messages.json(st.session_state.langchain_messages)
