# LangGraph Supervisor

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

## 실행 결과

"서울 날씨는?"로 입력후 결과를 확인합니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/419f8ec9-1533-414a-9fae-f790538623b7" />


"strawberry의 r의 갯수는?

<img width="700" alt="image" src="https://github.com/user-attachments/assets/4f988649-0c2a-499b-abca-1bc9ac6f11dd" />


"서울에서 부산을 거쳐서 제주로 가려고 합니다. 가는 동안의 날씨와 지역 맛집을 검색해서 추천해주세요."로 입력후 결과를 확인합니다. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/d8deb7ab-1b13-4ef4-a179-62f9044c981e" />
