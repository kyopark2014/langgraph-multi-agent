# LangGraph Swarm

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


## 실행 결과

"서울 날씨는?"이라고 질문하면 search agent에서 weather agent로 이동 후에 날씨 정보를 조회합니다.

<img src="https://github.com/user-attachments/assets/c7e1e998-aeb1-4bac-b98f-6de05bcc41b2" width="700">


"서울에서 부산을 거쳐서 제주로 가려고합니다. 가는 동안의 현재 온도와 지역 맛집 검색해서 추천해주세요."로 입력후 결과를 확인합니다.

이때의 결과를 보면 아래와 같이, 시작이 search agent이므로 weather agent로 transfer하고 날씨 정보를 수집합니다.

<img src="https://github.com/user-attachments/assets/a01d7922-cd73-4879-ba79-2da1f8d14f70" width="700">


날씨 정보를 모두 수집하면 다시 search agent로 전환한 후에 검색을 수행합니다.

<img src="https://github.com/user-attachments/assets/7de5a1a7-5201-4b9a-b7aa-b1e248615338" width="700">


최종적으로 아래와 같이 서울, 부산, 제주의 온도와 맛집에 대한 정보를 아래처럼 수집하였습니다.

<img src="https://github.com/user-attachments/assets/5295485c-6077-4e88-9065-69e3b6b1f185" width="700">

