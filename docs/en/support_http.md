# Without Any Agentic Framework

Why use the Agent SDKs and all these abstractions? If you want to take control of the foundation of LLM Agents,
in this AI era, you can always start from scratch and build your own "high-scrapers".

## Http

- use `tuner.as_oai_baseurl_apikey()` to obtain baseurl + apikey arguments

### Explain with examples

=== "Before Convertion"

    ```python
    # tuner to api key
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-1234567"

    # take out query
    query = workflow_task.task.main_query

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

    # use raw http requests (non-streaming) to get response
    response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": "fill_whatever_model", # Of course, this `model` field will be ignored.
                "messages": messages,
            },
            headers={
                "Authorization": f"Bearer {api_key}"
            }
    )
    final_answer = response.json()['choices'][0]['message']['content']
    ```

=== "After Convertion (`as_oai_baseurl_apikey`)"

    ```python
    # tuner to api key
    url_and_apikey = tuner.as_oai_baseurl_apikey()
    base_url = url_and_apikey.base_url
    api_key = url_and_apikey.api_key

    # take out query
    query = workflow_task.task.main_query

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

    # use raw http requests (non-streaming) to get response
    response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": "fill_whatever_model", # Of course, this `model` field will be ignored.
                "messages": messages,
            },
            headers={
                "Authorization": f"Bearer {api_key}"
            }
    )
    final_answer = response.json()['choices'][0]['message']['content']
    ```



!!! warning ""
    - when you are using the `tuner.as_oai_baseurl_apikey()` api, you must enable the following feature in the yaml configuration.

    ```yaml

    ajet:
        ...
        enable_experimental_interchange_server: True
        ...

    ```


