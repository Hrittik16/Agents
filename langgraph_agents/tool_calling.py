from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add a and b"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a"""
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide a by b 
    It can handle division by zero
    """
    try: 
        return a / b
    except ZeroDivisionError:
        return "Error: Division by zero"

model = ChatOpenAI(model="gpt-4o-mini")

# Tool creation
tools = [multiply, add, subtract, divide]

# Tool binding
model_with_tools = model.bind_tools(tools)

while True:
    user_query = input("Enter a query: ")
    if user_query.lower() == "q":
        break
    
    result = model_with_tools.invoke(input=user_query)
    
    # Extract tool name and tool args from result.tool_calls
    if hasattr(result, 'tool_calls') and result.tool_calls:
        tool_name = str(result.tool_calls[0]["name"])
        tool_args = str(result.tool_calls[0]["args"])
        tool_func = tool_name + "(" + tool_args + ")"
        result = eval(tool_func)
        print(f"Tool Used: {tool_name}\nResult: {result}")
    else:
        print("No tool calls found in the result")
    





