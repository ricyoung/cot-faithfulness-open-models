import asyncio
from cot_faithfulness.async_runner import _generate_one
from cot_faithfulness.config import MODEL_REGISTRY, Settings
from openai import AsyncOpenAI

settings = Settings()
aclient = AsyncOpenAI(api_key=settings.openrouter_api_key, base_url=settings.api.base_url, timeout=60)
model_config = MODEL_REGISTRY["nemotron-nano-9b"]

async def test():
    out = await _generate_one(aclient, model_config, settings, "What is 2+2? Answer A) 3 B) 4 C) 5 D) 6")
    print(f"Answer: {out['answer_text'][:100]}")
    print(f"Thinking tokens: {out['reasoning_tokens']}")
    print(f"Latency: {out['latency_seconds']:.1f}s")
    print("SUCCESS")

asyncio.run(test())
