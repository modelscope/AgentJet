

import asyncio
from agentscope_tuner.utils.robust_dashscope import RobustDashScopeChatModel


async def test_dashscope_api():
    """Test the RobustDashScopeChatModel by making a simple API call."""
    try:
        llm = RobustDashScopeChatModel("qwen-plus", stream=False)

        # Sample messages for a basic conversation
        messages = [
            {"role": "user", "content": "Hello! Can you tell me a short joke?"}
        ]

        # Call the model
        response = await llm(messages)

        # Print and verify the response
        print(response)


    except Exception as e:
        print(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(test_dashscope_api())
