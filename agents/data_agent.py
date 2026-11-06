import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from Models.schema import RouterSchema, DataAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst


llm = pick_llm("high")

llm_router = llm.with_structured_output(RouterSchema)


# ---------------------------- DATA AGENT GRAPH ---------------------------- #


def router_node(state:DataAgentSchema):

    if state.selected_dataset.startswith("table:") or state.selected_dataset == "database":
        state.route_response = "sql"
        return state
    if state.selected_dataset.startswith("file:"):
        state.route_response = "etl"
        return state

    latest_message = state.messages[-1].content if state.messages else ""
    latest_text = latest_message if isinstance(latest_message, str) else str(latest_message)
    is_chart_request = any(word in latest_text.lower() for word in ("chart", "graph", "plot", "visualize", "visualise"))
    explicitly_file_based = bool(re.search(r"(?:data/(?:extract|transform|uploads)/|\.(?:csv|json|parquet)\b|uploaded file|my file)", latest_text, re.IGNORECASE))
    if is_chart_request and not explicitly_file_based:
        state.route_response = "sql"
        return state

    from langchain_core.messages import SystemMessage
    instructions = [
        SystemMessage(content="Route questions that mention a file path, uploaded/extracted/transformed data, or asking to call/extract an API to 'etl'. This includes questions about data saved in data/extract, data/transform, or data/uploads. Route questions about tables in the configured PostgreSQL database to 'sql'. If ambiguous, use 'etl' when the user mentions a file and 'sql' when the user mentions database tables.")
    ]
    route_response_dict = llm_router.invoke(instructions + state.messages).model_dump() # type: ignore

    route_response = route_response_dict['answer']

    state.route_response = route_response

    return state

def etl_node(state:DataAgentSchema):

    message = state.messages[-1].content
    etl_messages = list(state.messages)
    if state.selected_dataset.startswith("file:"):
        file_path = state.selected_dataset.removeprefix("file:")
        etl_messages[-1] = HumanMessage(content=f"Use the selected data file `{file_path}` for this request. User question: {message}")

    response = etl_analyst.invoke(
        {"messages": etl_messages}
    )
    state.messages = state.messages + [response]

    return state

def sql_node(state:DataAgentSchema):

    message = state.messages[-1].content

    input_schema = {
        "messages": [],
        "user_question": f"{message}",
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": "",
        "selected_dataset": state.selected_dataset,
    }

    response = sql_analyst.invoke(input_schema)

    state.messages = state.messages + [response]

    return state




data_agent_graph = StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")

def route_edge(state: DataAgentSchema) -> str:
    if state.route_response == "sql":
        return "sql_node"
    elif state.route_response == "etl":
        return "etl_node"
    else:
        raise ValueError(f"Invalid route response: {state.route_response}")


data_agent_graph.add_conditional_edges("router_node", route_edge,
                                      {
                                          "sql_node": "sql_node",
                                          "etl_node": "etl_node"
                                      })

data_agent = data_agent_graph.compile()

if __name__ == "__main__":

    response = data_agent.invoke(
        {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
         "route_response": ""}
    )

    print(response)




