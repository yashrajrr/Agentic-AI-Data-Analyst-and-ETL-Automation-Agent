import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.schema import ETLAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic


#------------------------------------ AGENT TOOLS ------------------------------------#


@tool
def extract_load_tool(url:str, output_folder:str, format:str) -> str:
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()
    return etl_tools.extract_load(url, output_folder, format)


@tool
def transform_load_tool(input_file_path:str,output_folder:str,output_format:str, user_question:str) -> str:
    """
    This tool transforms the data from the specified file and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.

    """
    etl_tools = ETLTools()

    try:
        frame = etl_tools.read_frame(input_file_path)
        context = etl_tools.transform_load_context(input_file_path)
    except Exception as exc:
        return f"Failed to read input data: {exc}"

    output_folder = "data/transform"
    output_dir = etl_tools.DATA_DIR / "transform"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = output_format.lower().lstrip(".")
    if output_format not in {"csv", "json", "parquet"}:
        output_format = "csv"
    output_path = output_dir / f"transformed_data.{output_format}"
    
    llm = pick_llm("high")

    prompt = f"""
            You are a Python Data Analyst who uses Pandas to analyze data. 
            A pandas dataframe named df is already loaded from {input_file_path}.
            as per the user's question. Do not provide any explanation or comments, only
            the code should be provided. The code should be in a format that can be executed 
            in a Python environment with Pandas installed. 
            Don't write anything else than Pandas Code. \n
            
            Set a variable named result to the transformed dataframe. Do not read another file.
            Save result to the exact output_path variable, using the {output_format} format.
            Do not use imports, filesystem operations, or access anything except df, pd, output_path and normal dataframe methods.
            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {context}\n

        """

    response_message = llm.invoke(prompt)
    content = response_message.content
    if isinstance(content, list):
        content = content[0].get("text", str(content)) if isinstance(content[0], dict) else str(content[0])  # type: ignore
    elif not isinstance(content, str):
        content = str(content)

    # Optional Cleaning
    import re
    pandas_code = re.sub(r"^```(?:python)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE).strip()

    # Execute the Pandas code
    results = etl_tools.execute_code(pandas_code, frame, str(output_path))

    if results.startswith("Failed"):
        return results
    if not output_path.is_file():
        return f"Transformation did not create the required output file. Result: {results}"
    return f"The data is transformed and saved to {output_path} in {output_format} format. \n\n Pandas Code Executed: \n {pandas_code} \n\n Execution Result: \n {results}"


# Toolkit 
tools = [extract_load_tool, transform_load_tool]


@tool
def analyze_file_tool(input_file_path: str, user_question: str) -> str:
    """Analyze an existing CSV, JSON, or Parquet file under data/."""
    etl_tools = ETLTools()
    try:
        frame = etl_tools.read_frame(input_file_path)
    except Exception as exc:
        return f"Failed to read data file: {exc}"
    llm = pick_llm("medium")
    prompt = f"""Answer the user's question about this dataset. If calculation is needed, return only pandas Python code using a preloaded dataframe named df, assign the answer to result, and do not use imports or filesystem/network access. If the question only asks what is in the data, answer directly from the supplied sample and summary.\n\nQuestion: {user_question}\nColumns: {list(frame.columns)}\nRow count: {len(frame)}\nDtypes: {frame.dtypes.astype(str).to_dict()}\nSample rows: {frame.head(8).to_dict(orient='records')}\nNumeric summary: {frame.describe(include='all').fillna('').to_dict()}"""
    response = llm.invoke(prompt).content
    content = response if isinstance(response, str) else str(response)
    cleaned = re.sub(r"^```(?:python)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE).strip()
    if re.search(r"\b(result\s*=|df\[|df\.|print\s*\()", cleaned) and not cleaned.lower().startswith(("here", "the dataset", "based on", "this dataset")):
        result = etl_tools.execute_code(cleaned, frame)
        if not result.startswith("Failed"):
            return result
    return content


tools.append(analyze_file_tool)

llm = pick_llm("high")
llm_bind = llm.bind_tools(tools)


# ---------------------------------------- AGENT GRAPH ---------------------------------------- #

def llm_node(state:ETLAgentSchema):

    messages = state.messages

    prompt = f"""
            You are a Python Data Analyst who has access to tools that can extract and load, 
            transform, analyze and load data. You will be provided with a user's question
            and you would need to perform the right ETL operations as per the user's question. 
            For questions about an already extracted, uploaded, or transformed file, call analyze_file_tool.
            Use the exact file path the user gives, or the most recently referenced data file.
            For API extraction, save to data/extract and use the requested format.
            If the operation is performed then inform the user and end the coversation.
            Here's the chat history: {messages}\n
    """

    final_answer = llm_bind.invoke(prompt)

    state.messages = messages + [final_answer]

    return state


def tool_node(state:ETLAgentSchema):
    """
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    """

    tools_results = []

    tools_by_name = {tool.name: tool for tool in tools}

    tool_calls = state.messages[-1].tool_calls

    for tool_call in tool_calls:

        tool = tools_by_name[tool_call['name']]
        observation = tool.invoke(tool_call['args'])

        tools_results.append(ToolMessage(content=observation, tool_call_id = tool_call['id']))

    state.messages = state.messages + tools_results

    return state   


# Nodes & Edges
etl_analyst_graph = StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)

etl_analyst_graph.add_edge(START, "llm_node")

def is_tool_call(state:ETLAgentSchema):
    tool_calls = state.messages[-1].tool_calls

    if tool_calls:
        return "tool_node"
    else:
        return "end"

etl_analyst_graph.add_conditional_edges(
    "llm_node",is_tool_call,
    {
        "tool_node": "tool_node",
        "end": END
    }
)

etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()

if __name__ == "__main__":
    # Compile the Graph
    

    # Optional
    png_data = etl_analyst.get_graph().draw_mermaid_png()
    with open("etl_analyst_graph.png", "wb") as f:
        f.write(png_data)

    response = etl_analyst.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")]}
    )

#     response = etl_analyst.invoke(
#          {"messages":[HumanMessage(content=f"""
#             I want to transform the data stored in the 'c:\\Data_Agent\\data\\extract\\extracted_data.csv' file 
#             and save the transformed data in the 'c:\\Data_Agent\\data\\transform' folder in the csv format.
#             The transformation should filter the data to show bulbasaur pokemon only.
# """)]}
#     )    

    print(response)
