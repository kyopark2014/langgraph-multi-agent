# Multi-agent Supervisor (Router)

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

## Weather Agent

"서울과 제주 날씨를 비교해주세요."로 입력 후에 결과를 확인합니다.

![noname](https://github.com/user-attachments/assets/ac8cbc2b-e8d9-4e41-8f3b-58b5109f2d02)



## Code Agent 

"strawberry의 r의 갯수는?"라고 입력후 결과를 확인합니다. 

![image](https://github.com/user-attachments/assets/a8e9f8d1-53a1-45af-8c8d-53d18b45ac92)


## Search Agent

"강남역 맛집은?"으로 입력후 결과를 확인합니다.

![noname](https://github.com/user-attachments/assets/7903e6c5-c90c-48e3-a19c-7b03bf5d9ba6)
