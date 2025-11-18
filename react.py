"""
ReAct是一种Agent框架，这个框架允许Agent调用指定的工具。模型输出被要求包含Thought和Action，
其中Thought是模型进行推理的过程，Action是模型将要调用的工具。之后工具会返回一个Observation,
这个观察最终会被再次传入Agent，形成不断增长的上下文，直到模型认为其找到了最终答案。

模型输出格式主要依靠Prompt来约束，输出的提取主要依赖正则表达式

代码构建中几个思考：
1. Prompt中怎么形容输出格式是很重要的，比如希望输出的格式是[tool name] tool parameter，
大模型可能会不输出[]，或者不输出tool name，这可能是对[]没有进行解释。
2. 工具调用的时候需要传入参数，但是这个参数不一定是一个query这样的格式，可能是一个严格限定
的，比如要求输入日期，时间等等，这个约束不一定大模型能很好的跟随，所以要做好容错。
3. 手撕对于写正则表达式有一定要求。
4. 这里包含一个利用OpenAI接口的Agent实现，这类接口需要输入的prompt是特定格式，不能直接输入字符串。
5. 似乎最后只会把action和observation这一部分作为历史继续传入，thought则不会进行传入。
"""

# 思路来源于Hello-Agent:https://github.com/datawhalechina/hello-agents
# The implementation for ReAct
import os
import re
from serpapi import SerpApiClient
from dotenv import load_dotenv
from openai import OpenAI

ReActPrompt = """
You are an agent that should follow my instruction to finish the task.

You can use the tools that I assigned, each tools has name as key and description as value, you can read the description to understand what it can do:
Tools:{}

You output format should follow below pattern:
Thought: Your thought
Action: You have two choices
- `[Tool name] input`: the tool name and the corresponding input
- `[Finish] Final answer`: the final answer of question
- return [Finish] only when you think you find the answer to the question.
- Please make sure add `[]` to help me use regex match 

And the question and history is
question:{}
history:{}
"""

class ReActAgent:
    def __init__(self, llm_agent, tools, max_iter=5):
        self.llm_agent = llm_agent
        self.history = []
        self.tools = tools
        self.max_iter = max_iter
    
    def thinking(self, question):
        iters = 0 
        tools_desc = self.tools.get_description()
        self.history = []
        while iters < self.max_iter:
            print(f"--iter{iters}--")
            prompt = ReActPrompt.format(tools_desc, question, '\n'.join(self.history))
            message = [{"role": "user", "content": prompt}]
            response = self.llm_agent.generate(message)
            thought = self._parse_thought(response)
            print(f"[Thought]:\n{thought}\n")

            action, action_query = self._parse_action(response)
            self.history.append(action + action_query)

            if action.startswith('Finish'):
                print(f"[Final Answer]:\n{action_query}\n")
                break
            else:
                print(f"[Tool]: {action}")
                print(f"[Query]:\n {action_query}\n")
                tool = self.tools.get_tool(action)
                observation = tool(action_query)
                print(f"[Observation]:\n{observation}\n")
            
            self.history.append(observation)       
            iters += 1
        if iters == self.max_iter:
            print("The ReAct Framwork attain its max loop times.")

    def _parse_action(self, response):
        action = re.match(r"^.*Action.*\[(.*)\](.*)", response, re.DOTALL)
        if action:
            return action.group(1).strip(), action.group(2).strip()
        else:
            print(f"Error:Do not match action keyword in:\n {response}")
    
    def _parse_thought(self, response):
        thought = re.match(r"^.*Thought:\s+(.*)\s+Action", response, re.DOTALL)
        if thought:
            return thought.group(1).strip()
        else:
            print(f"Error:Do not match thought keyword in:\n {response}")

class HelloAgent:
    def __init__(self, base_url=None, base_model=None, api_key=None):
        api_key = api_key if api_key else os.environ.get("LLM_API_KEY")
        base_url = base_url if base_url else os.environ.get("LLM_BASE_URL")
        self.base_model = base_model if base_model else os.environ.get("LLM_MODEL_ID")
        timeout = os.environ.get("LLM_TIMEOUT", 60)
        if not all([self.base_model, api_key, base_url]):
            raise ValueError("The base model, api key and url should be defined in .env")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
    
    def generate(self, message, temperature=0):
        response = self.client.chat.completions.create(
            model=self.base_model,
            messages=message,
            stream=True,
            temperature=temperature
            )
        

        print("")
        chunk_response = []
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            chunk_response.append(content)
        print()
        return "".join(chunk_response)

class ToolRegister:
    def __init__(self):
        self.tool_description = {}
        self.tool_func = {}

    def register_tool(self, tool_name, tool_description, func):
        self.tool_description.update({tool_name: tool_description})
        self.tool_func.update({tool_name: func})

    def get_description(self):
        return self.tool_description

    def get_tool(self, tool_name):
        return self.tool_func[tool_name]
    
    def check_tools(self):
        for name, desc in self.tool_description.items():
            print(f"Tool Name: {name}")
            print(f"Tool description: {desc}")
    
def search(query):

    api = os.getenv('SERA_API_KEY')
    params = {
            "engine": "google",
            "q": query,
            "api_key": api,
            "gl": "us",  
            "hl": "en", 
        }
        
    client = SerpApiClient(params)
    results = client.get_dict()

    if "answer_box_list" in results:
        return "\n".join(results["answer_box_list"])
    if "answer_box" in results and "answer" in results["answer_box"]:
        return results["answer_box"]["answer"]
    if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
        return results["knowledge_graph"]["description"]
    if "organic_results" in results and results["organic_results"]:
        snippets = [
            f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
            for i, res in enumerate(results["organic_results"][:3])
        ]
        return "\n\n".join(snippets)
        
    return f"Error: not found info related to '{query}'"

if __name__ == '__main__':
    toolregister = ToolRegister()
    toolregister.register_tool('search', "This is google search, you can input an query" \
    " and receive the output from google search", search)

    load_dotenv()

    llm = HelloAgent()
    agent = ReActAgent(llm, toolregister)

    question = "Please tell me how to make a study plan for learning agent LLM."
    agent.thinking(question)



