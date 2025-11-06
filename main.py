from typing import List, Tuple, Union
from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser

load_dotenv()

# 1) 定义一个简单工具：给出字符串长度
@tool
def str_len(text: str) -> str:
    """Return the length (number of characters) of the input string."""
    return str(len(text))

TOOLS: List[Tool] = [Tool.from_function(str_len)]

def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool not found: {name}")

# 2) ReAct Prompt（单输入版）
REACT_TEMPLATE = """You are a helpful agent that uses tools to solve problems.

You have access to the following tools:
{tools}

When you need to use a tool, use the following format exactly:

Thought: <your reasoning>
Action: <one of [{tool_names}]>
Action Input: <the input to the action>

When you have the final answer, use:

Thought: <your reasoning>
Final Answer: <your answer>

Begin!

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(REACT_TEMPLATE).partial(
    tools=render_text_description(TOOLS),
    tool_names=", ".join([t.name for t in TOOLS]),
)

# 3) LLM（务必指定 model）
llm = ChatOpenAI(
    model="gpt-4o-mini",      # 换成你可用的模型
    temperature=0,
    stop=["\nObservation"]    # 让模型在要写 Observation 前停下，交给我们填观测
)

# 4) 组装代理：输入 → prompt → llm → 解析为 AgentAction/AgentFinish
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)

def run(question: str, max_iters: int = 8) -> str:
    intermediate_steps: List[Tuple[AgentAction, str]] = []

    for _ in range(max_iters):
        step: Union[AgentAction, AgentFinish] = agent.invoke({
            "input": question,
            "agent_scratchpad": intermediate_steps
        })
        print(step)  # 观察 ReAct 输出

        if isinstance(step, AgentFinish):
            return step.return_values.get("output", "")

        # AgentAction 分派工具
        tool_name = step.tool
        tool_input = step.tool_input
        tool_obj = find_tool_by_name(TOOLS, tool_name)

        # 执行工具并记录 Observation
        observation = tool_obj.func(str(tool_input))
        print(f"Observation: {observation}")
        intermediate_steps.append((step, str(observation)))

    # 兜底：迭代超限
    return "Reached max iterations without Final Answer."

if __name__ == "__main__":
    q = "What is the length of the word 'DOG'?"
    answer = run(q)
    print("Final Answer:", answer)
