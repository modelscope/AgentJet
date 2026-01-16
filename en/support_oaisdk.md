# Supported Agent Frameworks: AgentScope

This article introduce the way to convert different types of ways to convert your existing workflows into AgentJet workflows.


## AgentScope

1. use `tuner.as_raw_openai_sdk_client()` to create a openai SDK

2. use `tuner.as_oai_baseurl_apikey()` to override openai SDK's baseurl + apikey argument

### Explain with examples

=== "Before Convertion"

    ```python
    import openai
    client = openai.OpenAI(api_key='sk-123456')
    messages = [
        {
            "role": "system",
            "content": self.system_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]
    reply_message: ChatCompletion = await client.chat.completions.create(messages=messages)
    final_answer = reply_message.choices[0].message.content
    ```

=== "After Convertion (`as_raw_openai_sdk_client`)"

    ```python

    client = tuner.as_raw_openai_sdk_client()
    messages = [
        {
            "role": "system",
            "content": self.system_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]
    reply_message: ChatCompletion = await client.chat.completions.create(messages=messages)
    final_answer = reply_message.choices[0].message.content
    ```


=== "After Convertion (`as_oai_baseurl_apikey`)"

    ```python
    import openai
    url_and_apikey = tuner.as_oai_baseurl_apikey()
    base_url = url_and_apikey.base_url
    api_key = url_and_apikey.api_key

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {
            "role": "system",
            "content": self.system_prompt
        },
        {
            "role": "user",
            "content": query
        }
    ]
    reply_message: ChatCompletion = await client.chat.completions.create(messages=messages)
    final_answer = reply_message.choices[0].message.content
    ```




    !!! warning ""
        - when you are using the `tuner.as_oai_baseurl_apikey()` api, you must enable the following feature in the yaml configuration.

        ```yaml

        ajet:
            ...
            enable_experimental_interchange_server: True
            ...

        ```



