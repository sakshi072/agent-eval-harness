from eval_harness.adapters.base import TargetResponse, TargetAdapter
import httpx

class KnowledgeAgentAdapter(TargetAdapter):

    def __init__(self, base_url:str, timeout:float=60.0):
        self.base_url = base_url
        self.timeout = timeout

    async def ask(self, question:str, thread_id: str = None) -> TargetResponse:
        payload = {"message":question}
        if thread_id:
            payload["thread_id"] = thread_id

        async with httpx.AsyncClient(self.timeout) as client:
            response = await client.post(f"{self.base_url}/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        return TargetResponse(
            answer=data["message"],
            retrieved_contexts=[c["text"] for c in data.get("citations", []) if c.get("text")],
            tools_called=data.get("tools_used", []),
            thread_id=data.get("thread_id"),
            raw_response=data
        )