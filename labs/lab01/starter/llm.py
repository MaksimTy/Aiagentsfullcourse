import os
import json

from src.agent import LLM, Reply


class FakeLLM(LLM):
    """Детерминированная модель для тестов цикла: без сети и без денег.

    Сценарий: сначала пробует factorial с неверным аргументом (проверяем,
    что ошибка приходит как данные), затем с верным, затем отвечает.
    """
    model = "fake-deterministic"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, window: list[dict], tools: list[dict]) -> Reply:
        self.calls += 1
        in_tokens = sum(len(str(m)) for m in window) // 4
        if self.calls == 1:
            return Reply(tool_calls=[{"name": "factorial",
                                      "arguments": {"n": -5}}],
                         model=self.model, in_tokens=in_tokens, out_tokens=20,
                         cost_usd=in_tokens * 1.5e-7 + 20 * 6e-7)
        if self.calls == 2:
            return Reply(tool_calls=[{"name": "factorial",
                                      "arguments": {"n": 17}}],
                         model=self.model, in_tokens=in_tokens, out_tokens=20,
                         cost_usd=in_tokens * 1.5e-7 + 20 * 6e-7)
        last = ""
        for msg in reversed(window):
            if msg.get("role") == "tool":
                last = msg.get("content", "")
                break
        return Reply(text=f"Готово. Результат инструмента: {last}",
                     model=self.model, in_tokens=in_tokens, out_tokens=40,
                     cost_usd=in_tokens * 1.5e-7 + 40 * 6e-7)


# --------------------------------------------------------------------------- #
# Фиктивная модель, которая всегда просит инструмент
# --------------------------------------------------------------------------- #

class InfiniteToolCallLLM(LLM):
    """
    Фиктивная модель, которая всегда просит инструмент.
    
    Используется для тестирования того, что цикл не может выполниться
    больше max_steps раз.
    """
    model = "infinite-tool-call"
    
    def __init__(self, tool_name: str = "factorial"):
        self.tool_name = tool_name
        self.calls = 0
    
    def chat(self, window: list[dict], tools: list[dict]) -> Reply:
        self.calls += 1
        in_tokens = sum(len(str(m)) for m in window) // 4
        return Reply(
            tool_calls=[{"name": self.tool_name, "arguments": {"n": 5}}],
            model=self.model,
            in_tokens=in_tokens,
            out_tokens=20,
            cost_usd=in_tokens * 1.5e-7 + 20 * 6e-7
        )

class OpenAICompatLLM(LLM):
    """Реальный провайдер. Специально в 30 строк: адаптер, а не фреймворк."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI         # импорт внутри: не нужен для тестов
        self.client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        self.model = model or os.environ.get("AGENT_MODEL_MAIN", "")
        self.price_in = float(os.environ.get("PRICE_IN_PER_MTOK", "0.15"))
        self.price_out = float(os.environ.get("PRICE_OUT_PER_MTOK", "0.60"))

    def chat(self, window: list[dict], tools: list[dict]) -> Reply:
        resp = self.client.chat.completions.create(
            model=self.model, messages=window, tools=tools, temperature=0
        )
        msg = resp.choices[0].message
        calls = [
            {"name": c.function.name,
             "arguments": json.loads(c.function.arguments or "{}"),
             "id": c.id}
            for c in (msg.tool_calls or [])
        ]
        u = resp.usage
        return Reply(
            text=msg.content or "", tool_calls=calls, model=self.model,
            in_tokens=u.prompt_tokens, out_tokens=u.completion_tokens,
            cost_usd=u.prompt_tokens / 1e6 * self.price_in
                     + u.completion_tokens / 1e6 * self.price_out,
        )