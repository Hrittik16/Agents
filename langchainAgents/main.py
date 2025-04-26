from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv() # Load environment variables from .env file not from os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ResearchResponse(BaseModel):
    '''
    We can specify all of the fields that we want as
    output from our LLM call as long as our class inherits 
    from the pydantic basemodel.
    '''
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

'''
This parser will allow us to take the output of the LLM and parse
it into our ResearchResponse schema and then we can use the
schema as a normal python class.
'''
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a
            research paper. Answer the user query and use necessary
            tools. Wrap the output in this format and provide no
            other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("Search: ")

raw_response = agent_executor.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output"))
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, " Raw Response - ", raw_response)

