import asyncio
import itertools
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from tool_manager import load_tools
from langchain_core.tools import tool


# ------------------------- Progress bar -------------------------
async def progress(message="Выполнение"):
    try:
        for c in itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]):
            print(f"\r{message} {c}", end="", flush=True)
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print("\r" + " " * (len(message) + 4) + "\r", end="", flush=True)
        raise


# ------------------------- Agent State -------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ------------------------- Assistant -------------------------
class Assistant:
    def __init__(self, model_name: str = "qwen3:30b-a3b-instruct-2507-q4_K_M", temperature: float = 0.1):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.history = []

        # внутренние инструменты
        self.internal_tools = [
            self._make_chat_tool(),
            self._make_show_history_tool(),
            self._make_clear_history_tool(),
            self._make_list_tools_tool(),
            self._make_save_history_tool()
        ]

        # внешние инструменты
        self.external_tools = load_tools()
        self.tools = self.internal_tools + self.external_tools

        self.system_message = SystemMessage(content="""
            Ты — секретарь-помощник. Пользователь даёт команды. 
            Определи, какие инструменты нужно вызвать и в каком порядке.
            Если несколько задач — выполняй последовательно.
        """)

        self._build_graph()

    # ------------------------- Внутренние инструменты -------------------------
    def _make_chat_tool(self):
        @tool
        async def chat_tool() -> str:
            """Генерирует текстовый ответ через модель, используя историю"""
            messages = self.history[-10:]
            response = await self.llm.ainvoke(messages)
            content = getattr(response, "content", None)
            if isinstance(content, dict):
                return content.get("arguments", {}).get("message", "Пустой ответ")
            elif isinstance(content, str):
                return content
            return str(content) if content else "Пустой ответ"
        return chat_tool

    def _make_show_history_tool(self):
        @tool
        async def show_history() -> str:
            """Возвращает текст всей истории с пометкой отправителя"""
            lines = []
            for msg in self.history:
                if hasattr(msg, "content") and msg.content:
                    sender = "Вы" if isinstance(msg, HumanMessage) else "Ассистент"
                    lines.append(f"{sender}: {msg.content}")
            return "\n".join(lines) if lines else "История пустая"
        return show_history

    def _make_clear_history_tool(self):
        @tool
        async def clear_history() -> str:
            """Очищает историю сообщений"""
            self.history.clear()
            return "🧹 История очищена!"
        return clear_history

    def _make_list_tools_tool(self):
        @tool
        async def list_tools() -> str:
            """Возвращает список всех доступных инструментов"""
            return "\n".join([t.name for t in self.tools])
        return list_tools

    def _make_save_history_tool(self):
        @tool
        async def save_history_to_file(filename: str = "history.txt") -> str:
            """Сохраняет историю диалога в файл"""
            try:
                def write_file():
                    with open(filename, "w", encoding="utf-8") as f:
                        for msg in self.history:
                            if hasattr(msg, "content") and msg.content:
                                sender = "Вы" if isinstance(msg, HumanMessage) else "Ассистент"
                                f.write(f"{sender}: {msg.content}\n")
                await asyncio.to_thread(write_file)
                return f"История сохранена в {filename}"
            except Exception as e:
                return f"Ошибка при сохранении: {e}"
        return save_history_to_file

    # ------------------------- LangGraph -------------------------
    def _build_graph(self):
        def should_continue(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "continue"
            return "end"

        async def call_model(state: AgentState) -> AgentState:
            llm_with_tools = self.llm.bind_tools(self.tools)
            response = await llm_with_tools.ainvoke(state["messages"])
            if response:
                self.history.append(response)
                return {"messages": state["messages"] + [response]}
            return {"messages": state["messages"]}

        self.builder = StateGraph(AgentState)
        self.builder.add_node("model", call_model)
        self.builder.add_node("tools", ToolNode(tools=self.tools))
        self.builder.set_entry_point("model")
        self.builder.add_conditional_edges(
            "model", should_continue, {"continue": "tools", "end": END}
        )
        # Если охото чтобы модель что то делала с результатом работы инструмента
        #self.builder.add_edge("tools", "model")
        # Если не охото
        self.builder.add_edge("tools", END)
        self.graph = self.builder.compile()

    # ------------------------- Обработка запроса -------------------------
    async def handle_input(self, user_input: str):
        messages = [self.system_message, HumanMessage(content=user_input)]
        self.history += messages

        task = asyncio.create_task(progress("Думает"))
        try:
            result = await self.graph.ainvoke({"messages": self.history}, config={"recursion_limit": 5})
        except Exception as e:
            print(f"\nОшибка при обработке запроса: {e}")
            result = None
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            print("✅ Выполнено!\n")

        if result and "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, "content") and last_message.content:
                print(f"Ассистент: {last_message.content}\n")

    # ------------------------- Управление историей -------------------------
    async def clear_history(self):
        self.history.clear()
        print("🧹 История очищена!")

    async def get_history_text(self):
        lines = []
        for msg in self.history:
            if hasattr(msg, "content") and msg.content:
                sender = "Вы" if isinstance(msg, HumanMessage) else "Ассистент"
                lines.append(f"{sender}: {msg.content}")
        return "\n".join(lines)

    # ------------------------- Приветственное сообщение -------------------------
    def get_welcome_message(self) -> str:
        lines = ["Привет! Я твой помощник. Список доступных команд:"]
        for t in self.tools:
            lines.append(f" - {t.name}")
        return "\n".join(lines)
