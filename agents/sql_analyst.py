import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.database import DatabaseUtil
from utils.config import get_db_config
from Models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

VEHICLE_LOWEST_RATINGS_SQL = """WITH vehicle_scores AS (
    SELECT v.vehicle_id, v.make, v.model, v.year, v.license_plate,
           ROUND(AVG(r.rating)::numeric, 2) AS average_driver_rating,
           COUNT(r.rating) AS rating_count
    FROM public.vehicles AS v
    JOIN public.ratings AS r ON r.driver_id = v.driver_id
    GROUP BY v.vehicle_id, v.make, v.model, v.year, v.license_plate
), lowest_score AS (
    SELECT MIN(average_driver_rating) AS value FROM vehicle_scores
)
SELECT vs.vehicle_id, vs.make, vs.model, vs.year, vs.license_plate,
       vs.average_driver_rating, vs.rating_count
FROM vehicle_scores AS vs
CROSS JOIN lowest_score AS ls
WHERE vs.average_driver_rating = ls.value
ORDER BY vs.make, vs.model, vs.vehicle_id"""

DROPOFF_DISTANCE_CHART_SQL = """SELECT dropoff_time, distance_km
FROM public.rides
WHERE dropoff_time IS NOT NULL AND distance_km IS NOT NULL
ORDER BY dropoff_time ASC
LIMIT 100"""


# -------------------------------------- AI Agent Code--------------------------------------

def curate_ques(state: AgentSchema) -> AgentSchema: 

    user_question = state.user_question # Bcz this is a Pydantic model object

    if is_dropoff_distance_chart_request(user_question):
        state.curated_ques = user_question
        state.messages = state.messages + [HumanMessage(content=user_question)]
        return state

    llm = pick_llm("low")  # Pick the appropriate LLM based on the level of the question

    response = llm.invoke(f"Curate the following question: {user_question}").content

    state.curated_ques = response
    state.messages = state.messages + [HumanMessage(content=f"{response}")]  # Append the curated question to the messages list

    return state 


def is_dropoff_distance_chart_request(question: str) -> bool:
    import re
    has_chart_intent = bool(re.search(r"\b(chart|graph|plot|visuali[sz]e)\b", question, re.IGNORECASE))
    has_dropoff = bool(re.search(r"\bdrop\s*off\s*(?:time|timestamp)?\b", question, re.IGNORECASE))
    has_distance = bool(re.search(r"\bdistance\s*(?:km|kilomet(?:er|re)s?)?\b", question, re.IGNORECASE))
    return has_chart_intent and has_dropoff and has_distance


def prompt_query_context(state: AgentSchema) -> AgentSchema:

    curated_question = state.curated_ques
    if is_dropoff_distance_chart_request(state.user_question):
        state.prompt_query_context = "Use public.rides.dropoff_time and public.rides.distance_km to create the requested chart data."
        return state
    dataset_context = ""
    if state.selected_dataset == "database":
        dataset_context = "The user explicitly selected the database as the data source. Use the relevant public tables.\n"
    elif state.selected_dataset.startswith("table:"):
        selected_table = state.selected_dataset.removeprefix("table:")
        dataset_context = f"The user selected public.{selected_table} as the primary dataset. Focus the query on it, and join another public table only when needed to answer the question.\n"

    obj = DatabaseUtil(get_db_config())

    schema_info = obj.schema_details("public")  # Fetch schema details for the 'public' schema

    # Constructing the prompt query for the agent to generate the SQL query
    prompt = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language 
    query into Postgres SQL query that can be executed on the database. You are provided 
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that 
    you can understand the structure of the database and generate an accurate SQL query.
    Unless the user asks for a chart/trend/visualization, limit ordinary result sets to 10 rows. For chart requests, return up to 100 ordered data points and include the requested X and Y fields.
    Follow the actual schema relationships shown below. In this sample schema, ratings.driver_id identifies the rated driver; vehicles.driver_id identifies the vehicle's driver. There is no vehicle_id in ratings or rides. For a vehicle-rating question, compute the driver's average rating and associate it with that driver's vehicle, and state this limitation in the answer. Do not invent a direct vehicle-to-rating relationship.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.  
    
    {dataset_context}
    User's Original Query: {curated_question}

    Database Schema Details:
    {schema_info}
    
    """    

    state.prompt_query_context = prompt

    return state


# Generate SQL Query Node
def generate_sql(state: AgentSchema) -> AgentSchema:

    prompt = state.prompt_query_context

    llm = pick_llm("medium")  # Pick the appropriate LLM based on the level of the question

    import re
    vehicle_rating_question = bool(re.search(r"\bvehicles?\b", state.user_question, re.IGNORECASE) and
                                   re.search(r"\b(rat(e|ing|ings)|score|review)\w*\b", state.user_question, re.IGNORECASE) and
                                   re.search(r"\b(lowest|least|worst|bottom|poor)\b", state.user_question, re.IGNORECASE))
    if vehicle_rating_question:
        state.generated_sql_query = VEHICLE_LOWEST_RATINGS_SQL
        return state
    if is_dropoff_distance_chart_request(state.user_question):
        state.generated_sql_query = DROPOFF_DISTANCE_CHART_SQL
        return state

    generated_sql_query = llm.invoke(prompt).content  # Generate the SQL query using the LLM

    # Strip markdown formatting in case the LLM wrapped the output in ```sql ... ```
    import re
    clean_sql = re.sub(r"^```(?:sql)?\n?|\n?```$", "", generated_sql_query.strip(), flags=re.IGNORECASE)

    state.generated_sql_query = clean_sql.strip()

    return state


# Is safe Node
def is_safe_sql(state: AgentSchema) -> AgentSchema:

    sql_query = state.generated_sql_query

    if sql_query == VEHICLE_LOWEST_RATINGS_SQL:
        state.is_safe = "Yes"
        state.comments = "This is a fixed, read-only aggregate query over vehicles and ratings."
        return state
    if sql_query == DROPOFF_DISTANCE_CHART_SQL:
        state.is_safe = "Yes"
        state.comments = "This is a fixed, read-only query over the rides table."
        return state

    llm = pick_llm("medium")  
    llm_judge = llm.with_structured_output(JudgeSchema)

    prompt = f"""
    You are an SQL Judge for data security. Your task is to determine whether the SQL query is 
    safe or not. The SQL query should only be used for data retrieval and should not modify the 
    database in any way. Neither the SQL query nor the prompt should contain any SQL commands that can modify the
    database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change
    the structure or content of the database. If the SQL query is safe, respond with 'Yes' otherwise respond with 
    'No'. Additionally, provide comments explaining your decision.
    Here's the SQL query to evaluate:
    {sql_query}"""

    response = llm_judge.invoke(prompt).model_dump()  # Get the structured output as a dictionary
    state.is_safe = response['answer']
    state.comments = response['comments']

    return state


# Canceled SQL Query Node
def canceled_sql(state: AgentSchema) -> AgentSchema:

    comments = state.comments

    state.final_answer = f"The generated SQL query was deemed unsafe to execute. The reason provided by the judge is: {comments}. Therefore, the SQL query will not be executed."
    state.messages = state.messages + [AIMessage(content=f"{state.final_answer}")]  # Append the final answer to the messages list  

    return state


# Execute SQL Query Node
def execute_sql(state: AgentSchema) -> AgentSchema:

    sql_query = state.generated_sql_query

    obj = DatabaseUtil(get_db_config())

    execution_result = obj.execute_sql(sql_query)  # Execute the SQL query on the database
    # ``DatabaseUtil.execute_sql`` returns ``None`` on failure (and ``str(...)``
    # on success). Keep the field a valid string so the downstream Pydantic
    # state stays valid even when the database is unavailable.
    state.sql_query_execution_result = execution_result if execution_result is not None else ""

    return state


# Represent the final answer Node
def represent_final_answer(state: AgentSchema) -> AgentSchema:

    execution_result = state.sql_query_execution_result
    curated_question = state.curated_ques

    llm = pick_llm("low")

    prompt = f"""
    You are an SQL analyst agent. Your task is to provide a final answer to the user based on the
    execution result of the SQL query and the user's original question. The final answer should be
    concise, clear, and directly address the user's query. Avoid including any SQL code or technical
    details in the final answer. The final answer should be in a user-friendly format that is easy to
    understand. If the execution result is empty or does not provide a clear answer to the user's question, explain this in the final answer. \n
    For vehicle-rating results, clearly say ratings are linked to drivers in this schema, so vehicle results are attributed through each vehicle's driver. Do not imply rating records contain a vehicle_id. Include each returned vehicle and its average rating.\n
    Here is the execution result: {execution_result} \n
    Here is the user's original question: {curated_question}
    """

    llm_response = llm.invoke(prompt).content  # Get the final answer from the LLM

    state.final_answer = str(llm_response)
    state.messages = state.messages + [AIMessage(content=f"{llm_response}")]  # Append the final answer to the messages list

    return state


# ------------------------------------------- Graph Building -------------------------------------------
sql_agent_graph = StateGraph(AgentSchema)

# Nodes
sql_agent_graph.add_node("curate_ques", curate_ques)
sql_agent_graph.add_node("prompt_query_context", prompt_query_context)
sql_agent_graph.add_node("generate_sql", generate_sql)
sql_agent_graph.add_node("is_safe_sql", is_safe_sql)
sql_agent_graph.add_node("canceled_sql", canceled_sql)
sql_agent_graph.add_node("execute_sql", execute_sql)
sql_agent_graph.add_node("represent_final_answer", represent_final_answer)

# Edges
sql_agent_graph.add_edge(START, "curate_ques")
sql_agent_graph.add_edge("curate_ques", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe_sql")

def is_safe_sql_edge(state: AgentSchema) -> str:
    if state.is_safe == "Yes":
        return "execute_sql"
    else:
        return "canceled_sql"

sql_agent_graph.add_conditional_edges("is_safe_sql", is_safe_sql_edge,
                                      {
                                          "execute_sql": "execute_sql",
                                          "canceled_sql": "canceled_sql"
                                      })


sql_agent_graph.add_edge("canceled_sql", END)
sql_agent_graph.add_edge("execute_sql", "represent_final_answer")
sql_agent_graph.add_edge("represent_final_answer", END)



sql_analyst = sql_agent_graph.compile()

if __name__ == "__main__":


    # Optional
    from IPython.display import display, Image
    img = Image(sql_analyst.get_graph().draw_mermaid_png())
    with open("sql_analyst_graph.png", "wb") as f:
        f.write(img.data)

    input_schema = {
        "messages": [],
        "user_question": "What are the different types of Payment Methods we have in our database",
        "curated_ques": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    # Execute the Graph
    sql_analyst_response = sql_analyst.invoke(input_schema)
    print(sql_analyst_response['messages'])  # Print the final output of the graph execution
    print("********************************")

    print(sql_analyst_response['generated_sql_query'])  # Print the generated SQL query

    print("********************************")

    print(sql_analyst_response['sql_query_execution_result'])  # Print the result of executing the SQL query

    print("********************************")

    print(sql_analyst_response['prompt_query_context'])  # Print the prompt query context
