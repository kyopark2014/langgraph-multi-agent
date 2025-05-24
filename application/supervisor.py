import utils
import chat
import traceback
import tool_use

from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from typing import Literal
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.graph import START, END, StateGraph
from langgraph_supervisor import create_supervisor, create_handoff_tool

import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("supervisor")

####################### LangGraph #######################
# Chat Agent Executor
#########################################################
def create_collaborator(tools, name, st):
    logger.info(f"###### create_collaborator ######")

    chatModel = chat.get_chat(extended_thinking="Disable")
    model = chatModel.bind_tools(tools)

    class State(TypedDict): 
        # messages: Annotated[Sequence[BaseMessage], operator.add]
        messages: Annotated[list, add_messages]
        name: str

    tool_node = ToolNode(tools)

    def should_continue(state: State) -> Literal["continue", "end"]:
        logger.info(f"###### should_continue ######")

        logger.info(f"state: {state}")
        messages = state["messages"]    

        last_message = messages[-1]
        logger.info(f"last_message: {last_message.content}")

        if last_message.content:
            st.info(f"{name}: {last_message.content}")

        if last_message.tool_calls:
            for message in last_message.tool_calls:
                args = message['args']
                if chat.debug_mode=='Enable': 
                    if "code" in args:                    
                        state_msg = f"tool name: {message['name']}"
                        utils.status(st, state_msg)                    
                        utils.stcode(st, args['code'])
                    
                    elif chat.model_type=='claude':
                        state_msg = f"tool name: {message['name']}, args: {message['args']}"
                        utils.status(st, state_msg)

            logger.info(f"--- CONTINUE: {last_message.tool_calls[-1]['name']} ---")
            return "continue"
        
        #if not last_message.tool_calls:
        else:
            # logger.info(f"Final: {last_message.content}")
            logger.info(f"--- END ---")
            return "end"
           
    def call_model(state: State, config):
        logger.info(f"###### call_model ######")
        logger.info(f"state: {state['messages']}")

        last_message = state['messages'][-1]
        if isinstance(last_message, ToolMessage):
            logger.info(f"{last_message.name}: {last_message.content}")
            if chat.debug_mode=="Enable":
                st.info(f"{last_message.name}: {last_message.content}")
                
        if chat.isKorean(state["messages"][0].content)==True:
            system = (
                "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                f"당신의 역할은 {name}입니다."
                "당신의 역할에 맞는 답변만을 정확히 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."                
            )
        else: 
            system = (            
                "You are a conversational AI designed to answer in a friendly way to a question."
                "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            )

        for attempt in range(3):   
            logger.info(f"attempt: {attempt}")
            try:
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )
                chain = prompt | model
                    
                response = chain.invoke(state["messages"])
                logger.info(f"call_model response: {response}")

                # extended thinking
                if chat.debug_mode=="Enable":
                    chat.show_extended_thinking(st, response)

                if isinstance(response.content, list):            
                    for re in response.content:
                        if "type" in re:
                            if re['type'] == 'text':
                                logger.info(f"--> {re['type']}: {re['text']}")

                                status = re['text']
                                logger.info(f"status: {status}")
                                
                                status = status.replace('`','')
                                status = status.replace('\"','')
                                status = status.replace("\'",'')
                                
                                logger.info(f"status: {status}")
                                if status.find('<thinking>') != -1:
                                    logger.info(f"Remove <thinking> tag.")
                                    status = status[status.find('<thinking>')+11:status.find('</thinking>')]
                                    logger.info(f"status without tag: {status}")

                                if chat.debug_mode=="Enable":
                                    utils.status(st, status)
                                
                            elif re['type'] == 'tool_use':                
                                logger.info(f"--> {re['type']}: {re['name']}, {re['input']}")

                                if chat.debug_mode=="Enable":
                                    utils.status(st, f"{re['type']}: {re['name']}, {re['input']}")
                            else:
                                logger.info(re)
                        else: # answer
                            logger.info(response.content)
                break
            except Exception:
                response = AIMessage(content="답변을 찾지 못하였습니다.")

                err_msg = traceback.format_exc()
                logger.info(f"error message: {err_msg}")
                # raise Exception ("Not able to request to LLM")

        return {"messages": [response]}

    def buildChatAgent():
        workflow = StateGraph(State)

        workflow.add_node("agent", call_model)
        workflow.add_node("action", tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "action",
                "end": END,
            },
        )
        workflow.add_edge("action", "agent")

        return workflow.compile(name=name)
    
    return buildChatAgent()
    
reference_docs = []
contentList = []
image_url = []
isInitiated=False

def run_langgraph_supervisor(query, st):
    logger.info(f"###### run_supervisor ######")
    logger.info(f"query: {query}")

    global search_agent, stock_agent, supervisor_agent, weather_agent, code_agent, isInitiated
    if not isInitiated:
        # creater search agent
        search_agent = create_collaborator(
            [tool_use.search_by_tavily], 
            "search_agent", st
        )
        # creater stock agent
        stock_agent = create_collaborator(
            [tool_use.stock_data_lookup], 
            "stock_agent", st
        )
        # creater weather agent
        weather_agent = create_collaborator(
            [tool_use.get_weather_info], 
            "weather_agent", st
        )
        # creater code agent
        code_agent = create_collaborator(
            [tool_use.code_drawer, tool_use.code_interpreter], 
            "code_agent", st
        )

        agents = [search_agent, stock_agent, weather_agent, code_agent]

        class State(TypedDict): 
            messages: Annotated[list, add_messages]
            remaining_steps: 50

        workflow = create_supervisor(
            agents=agents,
            state_schema=State,
            model=chat.get_chat(extended_thinking="Disable"),
            prompt = (
                "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
                f"질문에 대해 충분한 정보가 모아질 때까지 다음의 agent를 선택하여 활용합니다. agents: {agents}"
                "모든 agent의 응답을 모아서, 충분한 정보를 제공합니다."
                "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            ),
            tools=[
                create_handoff_tool(
                    agent_name="search_agent", 
                    name="assign_to_search_expert", 
                    description="search internet or RAG to answer all general questions such as restronent"),
                create_handoff_tool(
                    agent_name="stock_agent", 
                    name="assign_to_stock_expert", 
                    description="retrieve stock trend"),
                create_handoff_tool(
                    agent_name="weather_agent", 
                    name="assign_to_weather_expert", 
                    description="earn weather informaton"),
                create_handoff_tool(
                    agent_name="code_agent", 
                    name="assign_to_code_expert", 
                    description="generate a code to solve a complex problem")
            ],
            supervisor_name="langgraph_supervisor",
            output_mode="full_history" # last_message full_history
        )        
        supervisor_agent = workflow.compile(name="superviser")
        isInitiated = True

    inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50
    }
    
    result = supervisor_agent.invoke({"messages": inputs}, config)
    logger.info(f"messages: {result['messages']}")
    
    length = len(result["messages"])
    for i in range(length):
        index = length-i-1
        message = result["messages"][index]
        # logger.info(f"message[{index}]: {message.content}")

        stop_reason = ""
        if "stop_reason" in message.response_metadata:
            stop_reason = message.response_metadata["stop_reason"]

        if isinstance(message, AIMessage) and message.content and stop_reason=="end_turn":
            msg = message.content
            break    
    logger.info(f"msg: {msg}")

    for i, doc in enumerate(reference_docs):
        logger.info(f"--> {i}: {doc}")
        
    reference = ""
    if reference_docs:
        reference = chat.get_references(reference_docs)

    msg = chat.extract_thinking_tag(msg, st)

    return msg, image_url, reference
