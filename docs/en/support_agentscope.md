# Supported Agent Frameworks: AgentScope

This article introduce the way to convert different types of ways to convert your existing workflows into AgentJet workflows.


## AgentScope

1. use `tuner.as_agentscope_model()` to override ReActAgent's model argument

2. use `tuner.as_oai_baseurl_apikey()` to override OpenAIChatModel's baseurl + apikey argument

### Explain with examples

=== "Before Convertion"

    ```python
    model = DashScopeChatModel(model_name="qwen-max", stream=False)  # 🛩️ change here
    agent_instance = ReActAgent(
       name=f"Friday",
       sys_prompt="You are a helpful assistant",
       model=model,
       formatter=DashScopeChatFormatter(),
    )
    ```

=== "After Convertion (`as_agentscope_model`)"

    ```python
    model = tuner.as_agentscope_model() # 🛩️ change here
    agent_instance = ReActAgent(
       name=f"Friday",
       sys_prompt="You are a helpful assistant",
       model=model,
       formatter=DashScopeChatFormatter(),
    )
    ```

=== "After Convertion (`as_agentscope_model`)"

    ```python
    url_and_apikey = tuner.as_oai_baseurl_apikey()
    base_url = url_and_apikey.base_url
    api_key = url_and_apikey.api_key    # the api key contain information, do not discard it
    model = OpenAIChatModel(
        model_name="whatever",
        client_args={"base_url": base_url},
        api_key=api_key,
    )
    self.agent = ReActAgent(
        name="math_react_agent", sys_prompt=system_prompt,
        model=model,  # ✨✨ compared with a normal agentscope agent, here is the difference!
        formatter=OpenAIChatFormatter(),
        toolkit=self.toolkit,
        memory=InMemoryMemory(), max_iters=2,
    )
    ```


!!! warning ""
    - when you are using the `tuner.as_oai_baseurl_apikey()` api, you must enable the following feature in the yaml configuration.

    ```yaml

    ajet:
        ...
        enable_experimental_reverse_proxy: True
        ...

    ```




### Explain with examples (Full Workflow Code)



=== "Full Code After Convertion (`as_agentscope_model`)"

    ```python
    import re
    from loguru import logger
    from agentscope.message import Msg
    from agentscope.agent import ReActAgent
    from agentscope.formatter import DashScopeChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.tool import Toolkit, execute_python_code
    from ajet import AjetTuner, Workflow, WorkflowOutput, WorkflowTask


    def extract_final_answer(result) -> str:
        """Extract the final answer from the agent's response."""
        try:
            if (
                hasattr(result, "metadata")
                and isinstance(result.metadata, dict)
                and "result" in result.metadata
            ):
                return result.metadata["result"]
            if hasattr(result, "content"):
                if isinstance(result.content, dict) and "result" in result.content:
                    return result.content["result"]
                return str(result.content)
            return str(result)
        except Exception as e:
            logger.warning(f"Extract final answer error: {e}. Raw: {result}")
            return str(result)


    system_prompt = """
    You are an agent specialized in solving math problems with tools.
    Please solve the math problem given to you.
    You can write and execute Python code to perform calculation or verify your answer.
    You should return your final answer within \\boxed{{}}.
    """


    class MathToolWorkflow(Workflow): # ✨✨ inherit `Workflow` class
        name: str = "math_agent_workflow"

        async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
            # run agentscope
            query = workflow_task.task.main_query
            self.toolkit = Toolkit()
            self.toolkit.register_tool_function(execute_python_code)
            self.agent = ReActAgent(
                name="math_react_agent", sys_prompt=system_prompt,
                model=tuner.as_agentscope_model(),  # ✨✨ compared with a normal agentscope agent, here is the difference!
                formatter=DashScopeChatFormatter(),
                toolkit=self.toolkit,
                memory=InMemoryMemory(), max_iters=2,
            )
            self.agent.set_console_output_enabled(False)
            msg = Msg("user", query, role="user")
            result = await self.agent.reply(msg)
            final_answer = extract_final_answer(result)

            # compute reward
            reference_answer = workflow_task.task.metadata["answer"].split("####")[-1].strip()
            match = re.search(r"\\boxed\{([^}]*)\}", final_answer)
            if match: is_success = (match.group(1) == reference_answer)
            else:     is_success = False
            return WorkflowOutput(reward=(1.0 if is_success else 0.0), metadata={"final_answer": final_answer})

    ```

=== "Full Code After Convertion (`as_agentscope_model`)"

    ```python
    import re
    from loguru import logger
    from agentscope.message import Msg
    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.model import OpenAIChatModel
    from agentscope.memory import InMemoryMemory
    from agentscope.tool import Toolkit, execute_python_code
    from ajet import AjetTuner, Workflow, WorkflowOutput, WorkflowTask


    def extract_final_answer(result) -> str:
        """Extract the final answer from the agent's response."""
        try:
            if (
                hasattr(result, "metadata")
                and isinstance(result.metadata, dict)
                and "result" in result.metadata
            ):
                return result.metadata["result"]
            if hasattr(result, "content"):
                if isinstance(result.content, dict) and "result" in result.content:
                    return result.content["result"]
                return str(result.content)
            return str(result)
        except Exception as e:
            logger.warning(f"Extract final answer error: {e}. Raw: {result}")
            return str(result)


    system_prompt = """
    You are an agent specialized in solving math problems with tools.
    Please solve the math problem given to you.
    You can write and execute Python code to perform calculation or verify your answer.
    You should return your final answer within \\boxed{{}}.
    """


    class MathToolWorkflow(Workflow): # ✨✨ inherit `Workflow` class
        name: str = "math_agent_workflow"

        async def execute(self, workflow_task: WorkflowTask, tuner: AjetTuner) -> WorkflowOutput:
            # run agentscope
            query = workflow_task.task.main_query
            self.toolkit = Toolkit()
            self.toolkit.register_tool_function(execute_python_code)

            url_and_apikey = tuner.as_oai_baseurl_apikey()
            base_url = url_and_apikey.base_url
            api_key = url_and_apikey.api_key    # the api key contain information, do not discard it
            model = OpenAIChatModel(
                model_name="whatever",
                client_args={"base_url": base_url},
                api_key=api_key,
            )
            self.agent = ReActAgent(
                name="math_react_agent", sys_prompt=system_prompt,
                model=model,  # ✨✨ compared with a normal agentscope agent, here is the difference!
                formatter=OpenAIChatFormatter(),
                toolkit=self.toolkit,
                memory=InMemoryMemory(), max_iters=2,
            )
            self.agent.set_console_output_enabled(False)
            msg = Msg("user", query, role="user")
            result = await self.agent.reply(msg)
            final_answer = extract_final_answer(result)

            # compute reward
            reference_answer = workflow_task.task.metadata["answer"].split("####")[-1].strip()
            match = re.search(r"\\boxed\{([^}]*)\}", final_answer)
            if match: is_success = (match.group(1) == reference_answer)
            else:     is_success = False
            return WorkflowOutput(reward=(1.0 if is_success else 0.0), metadata={"final_answer": final_answer})
    ```

