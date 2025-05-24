import streamlit as st 
import chat
import supervisor
import router
import swarm
import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

# title
st.set_page_config(page_title='Multi Agent', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "일상적인 대화": [
        "대화이력을 바탕으로 챗봇과 일상의 대화를 편안히 즐길수 있습니다."
    ],
    "Multi-agent Supervisor (Router)": [
        "Multi-agent Supervisor (Router)에 기반한 대화입니다. 여기에서는 Supervisor/Collaborators의 구조를 가지고 있습니다."
    ],
    "LangGraph Supervisor": [
        "LangGraph Supervisor를 이용한 Multi-agent Collaboration입니다. 여기에서는 Supervisor/Collaborators의 구조를 가지고 있습니다."
    ],
    "LangGraph Swarm": [
        "LangGraph Swarm를 이용한 Multi-agent Collaboration입니다. 여기에서는 Agent들 사이에 서로 정보를 교환합니다."
    ],
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Amazon Bedrock을 이용해 다양한 형태의 대화를 구현합니다." 
        "여기에서는 MCP를 이용해 RAG를 구현하고, Multi agent를 이용해 다양한 기능을 구현할 수 있습니다." 
        "또한 번역이나 문법 확인과 같은 용도로 사용할 수 있습니다."
        "주요 코드는 LangChain과 LangGraph를 이용해 구현되었습니다.\n"
        "상세한 코드는 [Github](https://github.com/kyopark2014/mcp)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")
    
    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["일상적인 대화", "Multi-agent Supervisor (Router)", "LangGraph Supervisor", "LangGraph Swarm"], index=2
    )   
    st.info(mode_descriptions[mode][0])
    
    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ("Nova Premier", 'Nova Pro', 'Nova Lite', 'Nova Micro', 'Claude 4 Opus', 'Claude 4 Sonnet', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=6
    )

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    #print('debugMode: ', debugMode)

    chat.update(modelName, debugMode)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")
    # logger.info(f"clear_button: {clear_button}")

st.title('🔮 '+ mode)

if clear_button==True:
    chat.initiate()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)
            st.markdown(message["content"])

display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. 편안한 대화를 즐기실수 있으며, 파일을 업로드하면 요약을 할 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    uploaded_file = None
    
    st.session_state.greetings = False
    chat.clear_chat_history()
    st.rerun()    
    
# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")

    with st.chat_message("assistant"):
        if mode == '일상적인 대화':
            stream = chat.general_conversation(prompt)            
            response = st.write_stream(stream)
            logger.info(f"response: {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})

            chat.save_chat_history(prompt, response)

        elif mode == "Multi-agent Supervisor (Router)":
            sessionState = ""
            with st.status("thinking...", expanded=True, state="running") as status:
                response = router.run_router_supervisor(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                chat.save_chat_history(prompt, response)       

        elif mode == "LangGraph Supervisor":
            sessionState = ""
            with st.status("thinking...", expanded=True, state="running") as status:
                response = supervisor.run_langgraph_supervisor(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                chat.save_chat_history(prompt, response)       

        elif mode == "LangGraph Swarm":
            sessionState = ""
            with st.status("thinking...", expanded=True, state="running") as status:
                response = swarm.run_langgraph_swarm(prompt, st)
                st.write(response)
                logger.info(f"response: {response}")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                chat.save_chat_history(prompt, response)       

        