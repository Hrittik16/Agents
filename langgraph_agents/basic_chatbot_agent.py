from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model(
    model="gpt-4o-mini", 
    )

class State(TypedDict):
    '''
    This is where we define the states that our model and agents will have access to.
    It defines the flow of data through the graph.
    We will keep track of all the messages in the conversation. 
    For e.g. assistant message, user message, assistant message, user message, etc.
    '''
    # Messages will be of type list, add_message from langgraph will add new messages to the list
    # Annotated[type, how_we_want_to_modify_the_type] 
    messages: Annotated[list, add_messages] 


# We will use the StateGraph class to build the graph that uses the State class
graph_builder = StateGraph(State)

def chatbot(state: State):
    '''
    This is one of the nodes in our graph.
    This function will take in a state of type State (i.e. the class we have defined)
    It will return a modification to the state i.e. the next state we'll have in our graph
    '''
    # We take all the messages from the state and pass them to the llm
    # We then add the response from the llm to the state
    ''' Because we are returning something that matches the State class,
    the message returned from the llm will be added to the state/list '''
    return {"messages": [llm.invoke(state["messages"])]}


# Inorder to use the chatbot node we have to register it with the graph builder
graph_builder.add_node("chatbot", chatbot)

# For langgraph agents the nodes start with START and end with END
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Run the graph
graph = graph_builder.compile()

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        state = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
        last_message = state["messages"][-1].content
        print(f"Assistant: {last_message}")
    except:
        print("Error!")
        break
