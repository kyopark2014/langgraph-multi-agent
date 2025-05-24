# LangGraph로 Multi Agent의 구현

여기서는 LangGraph를 이용헤 multi agent를 구현하는 3가지 방법에 대해 설명합니다. 

## Multi-agent Supervisor (Router)

여기에서는 multi-agent supervisor (Router) 방식으로 multi-agent collaboration을 구현하는것에 대해 설명합니다.

아래와 같이 state와 router를 정의합니다. 

```python
class State(MessagesState):
    next: str
    answer: str

members = ["search_agent", "code_agent", "weather_agent"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["search_agent", "code_agent", "weather_agent", "FINISH"]
```

이때의 supervisor node는 아래와 같습니다.

```python
def supervisor_node(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    structured_llm = llm.with_structured_output(Router, include_raw=True)
    
    chain = prompt | structured_llm
                
    messages = state['messages']
  
    response = chain.invoke({"messages": messages})
    parsed = response.get("parsed")
  
    goto = parsed["next"]
    if goto == "FINISH":            
        goto = END

    return Command(goto=goto, update={"next": goto})
```

아래와 같이 agent들을 정의합니다.

```python
search_agent = create_collaborator(
    [tool_use.search_by_tavily, tool_use.search_by_knowledge_base], 
    "search_agent", st
)

weather_agent = create_collaborator(
    [tool_use.get_weather_info], 
    "weather_agent", st
)

code_agent = create_collaborator(
    [tool_use.repl_coder, tool_use.repl_drawer], 
    "code_agent", st
)
```

Search, code, weather agent들을 정의합니다.

```python
def search_node(state: State) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="search_agent")
            ]
        },
        goto = "supervisor",
    )

def code_node(state: State) -> Command[Literal["supervisor"]]:
    result = code_agent.invoke(state)

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="code_agent")
            ]
        },
        goto = "supervisor",
    )

def weather_node(state: State) -> Command[Literal["supervisor"]]:
    result = weather_agent.invoke(state)

    return Command(
        update={
            "messages": [
                AIMessage(content=result["messages"][-1].content, name="weather_agent")
            ]
        },
        goto = "supervisor",
    )
```

아래와 같이 workflow를 정의합니다.

```python
def build_graph():
    workflow = StateGraph(State)
    workflow.add_edge(START, "supervisor")
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("search_agent", search_node)
    workflow.add_node("code_agent", code_node)
    workflow.add_node("weather_agent", weather_node)

    return workflow.compile()
````

이제 아래와 같이 실행할 수 있습니다.

```python
inputs = [HumanMessage(content=query)]
config = {
    "recursion_limit": 50
}    
result = app.invoke({"messages": inputs}, config)

msg = result['messages'][-1].content
```

### 실행결과 
#### Weather Agent

"서울과 제주 날씨를 비교해주세요."로 입력 후에 결과를 확인합니다.

![noname](https://github.com/user-attachments/assets/ac8cbc2b-e8d9-4e41-8f3b-58b5109f2d02)



#### Code Agent 

"strawberry의 r의 갯수는?"라고 입력후 결과를 확인합니다. 

![image](https://github.com/user-attachments/assets/a8e9f8d1-53a1-45af-8c8d-53d18b45ac92)


#### Search Agent

"강남역 맛집은?"으로 입력후 결과를 확인합니다.

![noname](https://github.com/user-attachments/assets/7903e6c5-c90c-48e3-a19c-7b03bf5d9ba6)




## LangGraph Supervisor

[LangGraph Multi-Agent Supervisor](https://github.com/langchain-ai/langgraph-supervisor-py)을 이용하면 multi-agent collaboration을 구현합니다.

이를 위해 langgraph-supervisor을 설치합니다.

```text
pip install langgraph-supervisor
```

동작은 아래와 같습니다.

![image](https://github.com/user-attachments/assets/b7ec2913-804b-4b4a-a1a9-d972ddb9a591)


아래와 같이 collaborator들을 준비합니다.

```python
search_agent = create_collaborator(
    [search_by_tavily, search_by_knowledge_base], 
    "search_agent", st
)
stock_agent = create_collaborator(
    [stock_data_lookup], 
    "stock_agent", st
)
weather_agent = create_collaborator(
    [get_weather_info], 
    "weather_agent", st
)
code_agent = create_collaborator(
    [code_drawer, code_interpreter], 
    "code_agent", st
)

def create_collaborator(tools, name, st):
    chatModel = chat.get_chat(extended_thinking="Disable")
    model = chatModel.bind_tools(tools)
    tool_node = ToolNode(tools)

    class State(TypedDict): 
        messages: Annotated[list, add_messages]
        name: str

    def should_continue(state: State) -> Literal["continue", "end"]:
        messages = state["messages"]    
        last_message = messages[-1]
        if last_message.tool_calls:
            return "continue"        
        else:
            return "end"
           
    def call_model(state: State, config):
        last_message = state['messages'][-1]
                
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

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt | model
        response = chain.invoke(state["messages"])
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
```

Supervisor agent는 아래와 같이 생성합니다.

```python
from langgraph_supervisor import create_supervisor, create_handoff_tool

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
```

아래와 같이 실행합니다.

```python
inputs = [HumanMessage(content=query)]
    config = {
        "recursion_limit": 50
    }
    
    result = supervisor_agent.invoke({"messages": inputs}, config)
    msg = result["messages"][-1].content
```

### 실행 결과

"서울 날씨는?"로 입력후 결과를 확인합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/419f8ec9-1533-414a-9fae-f790538623b7" />


"strawberry의 r의 갯수는?

<img width="700" alt="image" src="https://github.com/user-attachments/assets/4f988649-0c2a-499b-abca-1bc9ac6f11dd" />


"서울에서 부산을 거쳐서 제주로 가려고 합니다. 가는 동안의 날씨와 지역 맛집을 검색해서 추천해주세요."로 입력후 결과를 확인합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/d8deb7ab-1b13-4ef4-a179-62f9044c981e" />





## LangGraph Swarm

[LangGraph Multi-Agent Swarm](https://github.com/langchain-ai/langgraph-swarm-py/tree/main)와 같이 때로는 agent가 직접 메시지 및 history를 공유함으로써 multi-agent를 구성하는것이 좋을수도 있습니다. [Swarm](https://github.com/openai/swarm)은 agent간에 handoff가 가능합니다. 아래에서는 [LangGraph Swarm](https://www.youtube.com/watch?v=iqXn6Oiis4Q)을 참조하여 multi-agent를 구성하는 것을 설명합니다.

<img src="https://github.com/user-attachments/assets/8f719734-9644-4d26-990f-b771c999afc5" width="700">


Swarm agent가 search와 weather agent를 가졌을 때의 결과입니다.

<img src="https://github.com/user-attachments/assets/4f3fde50-9a73-40f1-88e6-d839c2f2ce8a" width="400">

Swarm agent에 search, code, weather agent의 구조를 가졌을때의 모습니다. Agent들이 mash 형태로 서로 연결되어 있다면 agent들이 증가할 때마다 복잡도가 높아집니다.

<img src="https://github.com/user-attachments/assets/80c5f0c3-c849-4025-b482-cbfc882c3020" width="700">


아래와 같이 LangGraph의 Swarm을 설치합니다.

```text
pip install langgraph-swarm
```

search agent가 weather agent로 이동하기 위해서, create_handoff_tool로 transfer_to_search_agent을 정의합니다. 마찬가지로 weather agent에서 search agent로 이동하기 위한 transfer_to_weather_agent을 아래와 같이 정의합니다.

```python
from langgraph_swarm import create_handoff_tool

transfer_to_search_agent = create_handoff_tool(
    agent_name="search_agent",
    description="Transfer the user to the search_agent for search questions related to the user's request.",
)
transfer_to_weather_agent = create_handoff_tool(
    agent_name="weather_agent",
    description="Transfer the user to the weather_agent to look up weather information for the user's request.",
)
```

이제 collaborator로 search와 weather agent를 정의합니다. search agent는 tavily 검색과 완전관리형 RAG 서비스인 search_by_knowledge_base를 가지고 있고, weather에서 search로 이동하기 위한 transfer_to_search_agent가 있습니다. weather agent는 날씨 검색을 위한 get_weather_info라는 tool과 weather에서 search agent로 전환을 위한 transfer_to_search_agent을 가지고 있습니다.

```python
# creater search agent
search_agent = create_collaborator(
    [search_by_tavily, search_by_knowledge_base, transfer_to_weather_agent], 
    "search_agent", st
)

# creater weather agent
weather_agent = create_collaborator(
    [get_weather_info, transfer_to_search_agent], 
    "weather_agent", st
)
```

이제 creat_swarm을 이용하여 swarm_agent을 준비합니다. swarm_agent는 search와 weather agent들을 가지고 있고, default로 search agent를 이용합니다. 

```python
from langgraph_swarm import create_swarm

swarm_agent = create_swarm(
    [search_agent, weather_agent], default_active_agent="search_agent"
)
langgraph_app = swarm_agent.compile()
```

아래와 같이 swarm_agent를 invoke하여 결과를 얻습니다.

```python
inputs = [HumanMessage(content=query)]
config = {
    "recursion_limit": 50
}

result = langgraph_app.invoke({"messages": inputs}, config)
```


### 실행 결과

"서울 날씨는?"이라고 질문하면 search agent에서 weather agent로 이동 후에 날씨 정보를 조회합니다.

<img src="https://github.com/user-attachments/assets/c7e1e998-aeb1-4bac-b98f-6de05bcc41b2" width="700">


"서울에서 부산을 거쳐서 제주로 가려고합니다. 가는 동안의 현재 온도와 지역 맛집 검색해서 추천해주세요."로 입력후 결과를 확인합니다.

이때의 결과를 보면 아래와 같이, 시작이 search agent이므로 weather agent로 transfer하고 날씨 정보를 수집합니다.

<img src="https://github.com/user-attachments/assets/a01d7922-cd73-4879-ba79-2da1f8d14f70" width="700">


날씨 정보를 모두 수집하면 다시 search agent로 전환한 후에 검색을 수행합니다.

<img src="https://github.com/user-attachments/assets/7de5a1a7-5201-4b9a-b7aa-b1e248615338" width="700">


최종적으로 아래와 같이 서울, 부산, 제주의 온도와 맛집에 대한 정보를 아래처럼 수집하였습니다.

<img src="https://github.com/user-attachments/assets/5295485c-6077-4e88-9065-69e3b6b1f185" width="700">

