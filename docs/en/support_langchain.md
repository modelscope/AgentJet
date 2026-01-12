# Supported Agent Frameworks: AgentScope

This article introduce the way to convert different types of ways to convert your existing workflows into AgentJet workflows.


## AgentScope

1. use `tuner.as_oai_baseurl_apikey()` to override OpenAIChatModel's baseurl + apikey argument

### Explain with examples

=== "Before Convertion"

    ```python
    from langchain_openai import ChatOpenAI




    # create openai model
    llm = ChatOpenAI(
        model="gpt-5",
    )
    agent=create_agent(
        model=llm,
        system_prompt=self.system_prompt,
    )

    # take out query
    query = workflow_task.task.main_query

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
    })
    ```

=== "After Convertion (`as_oai_baseurl_apikey`)"

    ```python
    from langchain_openai import ChatOpenAI

    url_and_apikey = tuner.as_oai_baseurl_apikey()
    base_url = url_and_apikey.base_url
    api_key = url_and_apikey.api_key

    llm = ChatOpenAI(
        model="whatever",
        base_url=base_url,
        api_key=lambda:api_key,
    )
    agent = create_agent(
        model=llm,
        system_prompt=self.system_prompt,
    )

    # take out query
    query = workflow_task.task.main_query

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ],
    })
    ```



!!! warning ""
    - when you are using the `tuner.as_oai_baseurl_apikey()` api, you must enable the following feature in the yaml configuration.

    ```yaml

    ajet:
        ...
        enable_experimental_reverse_proxy: True
        ...

    ```


