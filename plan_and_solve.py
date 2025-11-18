"""
ReAct框架有一个问题是可能陷入局部最优，也就是没有一个长期规划。因此一种新的框架Plan&Solve agent
框架被提出，这个框架采取了两步解决问题，先让agent生成一个plan，然后再基于plan进行问题的解决。在
一些数学题这类的连贯性要求比较高的问题上表现比较好。

代码构建中的思考：
1. 这里history我只放入了每一步的思考输出，但是hello-agent是把每个步骤的任务和对应的输出都传入的
在最后的问题解决上似乎也没什么区别。
2. 在对于执行器的prompt中，我并未传入plan，只传入了当前任务，结果仍然是可以正确输出的。
"""

# 思路来源于Hello-Agent:https://github.com/datawhalechina/hello-agents
# The implementation of plan and solve agent
import os
from openai import OpenAI
import ast
import re
from dotenv import load_dotenv

PLANNER_PROMPT = """
You are an smart agent that follow my instruction to build a plan for solving given question.

The question is:
{question}

You should return a response having following strcuture
```python
[step1, step2,...]
```
The '```python' and '```' is needed for identify output. And step1, step2, ..., which
 should be str type, is the plan given by you to solve the question. These steps are
 contained in a python list.
"""

EXECUTE_PROMPT = """
You are an smart agent that follow my instruction. The role of you is to finish the
given task and return output.

The question is:
{question}

The past output is:
{history}

In this step, you have task
{task}

After understand tasks, you should output your thought in this step with below format:
```[Thought] your thought```
Please keep '```' for convenient regex match.
"""

load_dotenv()


class Planner:
    def __init__(self, llm_agent):
        self.agent = llm_agent

    def plan(self, question):
        prompt = PLANNER_PROMPT.format(question=question)
        prompt = [{"role":"user", "content": prompt}]

        response = self.agent.generate(prompt)

        plan_str = response.split("```python")[1].split("```")[0]
        plan = ast.literal_eval(plan_str)
        return plan

class Executor:
    def __init__(self, llm_agent):
        self.agent = llm_agent
        self.history = []

    def execute(self, question, plan):
        self.history = []
        for i, task in enumerate(plan):
            print(f'--iter{i}--')
            prompt = EXECUTE_PROMPT.format(
                question=question,
                history='\n'.join(self.history),
                task=task,
                )
            prompt = [{'role':'user', 'content': prompt}]
            response = self.agent.generate(prompt)
            thought = self._parse_thought(response)
            print(f"[Task]:\n{task}")
            if i < len(plan) - 1:
                print(f"\n{thought}\n")
                self.history.append(thought)
            else:
                print(f"[Final Answer]:\n{thought}\n")



    def _parse_thought(self, output):
        thought = re.match(r".*```(.*)```.*", output, re.DOTALL)
        return thought.group(1)

class PaSAgent:
    def __init__(self, planner, executor):
        self.planner = planner
        self.executor = executor
    
    def plan_solve(self, question):
        plan = self.planner.plan(question)
        self.executor.execute(question, plan)

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
        
        chunk_response = []
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            chunk_response.append(content)
        return "".join(chunk_response)


if __name__ == '__main__':
    question = "A fruit store sold 15 apples on Monday. The number of apples sold on Tuesday was twice that of Monday. The number sold on Wednesday was 5 less than that on Tuesday. How many apples were sold in total over these three days?"
    llm_agent = HelloAgent()
    planner = Planner(llm_agent=llm_agent)
    executor = Executor(llm_agent=llm_agent)

    psa_agent = PaSAgent(planner, executor)
    psa_agent.plan_solve(question)
