"""
Модуль 01. Минимальное ядро агента: цикл, бюджеты, стоп-условия, трейс.

Запуск без ключей и без сети (детерминированная фиктивная модель):
    python 01-agent-basics/code/agent.py

Запуск с реальной моделью:
    export OPENAI_API_KEY=...            # или совместимый провайдер
    python 01-agent-basics/code/agent.py --live --task "посчитай 17!"

Принципы, заложенные в этот файл:
  1. Границы проверяются ДО траты денег.
  2. Ошибка инструмента - это данные, а не исключение.
  3. history (правда о прогоне) и window (что отправлено модели) разделены.
  4. Каждый выход из цикла имеет код причины.
  5. Трейс пишется всегда, в JSONL, по одной записи на строку.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from src.agent import (
    Budget, 
    Task, 
    Tool, 
    ToolResult, 
    ToolRegistry, 
    Trace,
    LLM, 
    Reply)
from tools.env import load_env_file

load_env_file()



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


# --------------------------------------------------------------------------- #
# Сборка окна: единственное место, куда позже встроится компакция (модуль 04)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """Ты инженерный агент. Работай по шагам.

Правила:
1. Если нужно действие - вызывай инструмент, не описывай его словами.
2. Результат инструмента - единственный источник фактов. Не выдумывай.
3. Если инструмент вернул ОШИБКУ, не повторяй тот же вызов с теми же
   аргументами: измени аргументы или выбери другой путь.
4. Когда задача решена, дай финальный ответ без вызова инструментов.
5. Если данных не хватает и получить их нечем - скажи об этом прямо.
"""


def build_window(task: Task, history: list[dict], warning: str | None) -> list[dict]:
    """Пока просто: system + вся история. В модуле 04 здесь появится
    компакция, приоритизация и пометка недоверенного контента."""
    window: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if warning:
        window.append({"role": "system", "content": warning})
    window.extend(history)
    return window


def observation(text: str) -> dict[str, str]:
    return {"role": "user", "content": f"[наблюдение] {text}"}


def tool_message(call: dict, res: ToolResult) -> dict[str, Any]:
    return {"role": "tool", "name": call["name"],
            "tool_call_id": call.get("id", call["name"]),
            "content": res.to_model()}


# --------------------------------------------------------------------------- #
# Цикл
# --------------------------------------------------------------------------- #

EMPTY_LIMIT = 2          # сколько пустых шагов терпим
REPEAT_LIMIT = 3         # сколько одинаковых вызовов терпим


def run(task: Task, tools: ToolRegistry, llm: LLM, trace: Trace) -> dict[str, Any]:
    budget = Budget(task=task)
    history: list[dict] = [{"role": "user", "content": task.prompt}]
    seen: list[str] = []

    while True:
        # 1. Границы - до траты денег
        reason = budget.exceeded()
        if reason:
            return finish(reason, history, budget, trace)

        # 2. Наблюдение
        window = build_window(task, history, budget.soft_warning())

        # 3. Решение
        budget.steps += 1
        t0 = time.monotonic()
        reply = llm.chat(window, tools.schemas())
        latency_ms = int((time.monotonic() - t0) * 1000)
        budget.tokens += reply.in_tokens + reply.out_tokens
        budget.cost_usd += reply.cost_usd
        trace.step(budget.steps, window, reply, budget, latency_ms)

        # 4a. Пустой шаг
        if not reply.tool_calls and not reply.text.strip():
            budget.empty_steps += 1
            if budget.empty_steps > EMPTY_LIMIT:
                return finish("no_progress", history, budget, trace)
            history.append(observation(
                "Пустой ответ. Выбери инструмент или дай итог."))
            continue

        # 4b. Финальный ответ - проверяем постусловие
        if not reply.tool_calls:
            ok, note = verify(task, reply.text)
            if ok:
                code = "done" if task.postcondition else "done_unverified"
                return finish(code, history, budget, trace, answer=reply.text)
            history.append(observation(f"Постусловие не выполнено: {note}"))
            continue

        history.append({"role": "assistant", "content": reply.text or None,
                        "tool_calls": reply.tool_calls})

        # 5. Действия
        for call in reply.tool_calls:
            fp = tools.fingerprint(call)
            if seen.count(fp) >= REPEAT_LIMIT:
                return finish("repeated_action", history, budget, trace)
            seen.append(fp)

            t1 = time.monotonic()
            res = tools.invoke(call)
            trace.tool(budget.steps, call, res, int((time.monotonic() - t1) * 1000))
            history.append(tool_message(call, res))


def verify(task: Task, answer: str) -> tuple[bool, str]:
    if task.postcondition is None:
        return True, "постусловия нет"
    try:
        return task.postcondition(answer)
    except Exception as exc:                          # noqa: BLE001
        return False, f"проверка упала: {exc}"


def finish(reason: str, history: list[dict], budget: Budget, trace: Trace,
           answer: str | None = None) -> dict[str, Any]:
    trace.finish(reason, budget, answer)
    return {
        "reason": reason,
        "answer": answer,
        "steps": budget.steps,
        "tokens": budget.tokens,
        "cost_usd": round(budget.cost_usd, 6),
        "seconds": round(budget.elapsed(), 2),
        "history_len": len(history),
        "trace": str(trace.path),
    }


# --------------------------------------------------------------------------- #
# Демонстрационные инструменты
# --------------------------------------------------------------------------- #

def _factorial(n: int) -> str:
    print(_factorial.__name__, n)
    n = int(n)
    if n < 0:
        raise ValueError("факториал определён только для n >= 0")
    if n > 2000:
        raise ValueError("n слишком велико, предел 2000")
    return str(math.factorial(n))


def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        siblings = sorted(x.name for x in (p.parent if p.parent.exists()
                                           else Path(".")).iterdir())[:10]
        raise FileNotFoundError("файла нет; рядом: " + ", ".join(siblings))
    return p.read_text(encoding="utf-8", errors="replace")


DEMO_TOOLS = ToolRegistry([
    Tool(
        name="factorial",
        description=("Точно вычисляет факториал целого неотрицательного n. "
                     "Возвращает десятичную запись числа. Предел n = 2000. "
                     "Используй вместо самостоятельного счёта в уме."),
        parameters={
            "type": "object",
            "properties": {"n": {"type": "integer", 
                                 "minimum": 0,
                                 "maximum": 2000,
                                 "description": "неотрицательное целое"}},
            "required": ["n"],
        },
        fn=_factorial,
    ),
    Tool(
        name="read_file",
        description=("Читает текстовый файл целиком и возвращает содержимое. "
                     "Путь относительно корня проекта. Если файла нет, "
                     "вернёт ошибку со списком соседних файлов."),
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string",
                                    "description": "относительный путь"}},
            "required": ["path"],
        },
        fn=_read_file,
    ),
])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Модуль 01: минимальное ядро агента")
    ap.add_argument("--task", default="Посчитай 17 факториал через инструмент.")
    ap.add_argument("--live", action="store_true",
                    help="использовать реального провайдера вместо FakeLLM")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--max-cost", type=float, default=0.50)
    args = ap.parse_args(argv)
    
   
    args.live = 'live' if args.live else os.environ.get('AGENT_MODE')
    print(args.live)

    run_id = "r-" + hashlib.sha256(
        (args.task + str(time.time())).encode()).hexdigest()[:6]
    trace = Trace(run_id)
    llm = {
        'live': OpenAICompatLLM(), 
        'fake':  FakeLLM() }.get(args.live)

    task = Task(prompt=args.task, max_steps=args.max_steps,
                max_cost_usd=args.max_cost)

    result = run(task, DEMO_TOOLS, llm, trace)

    print("=" * 62)
    print(f"run_id:   {run_id}")
    print(f"причина:  {result['reason']}")
    print(f"шаги:     {result['steps']}")
    print(f"токены:   {result['tokens']}")
    print(f"стоимость:{result['cost_usd']} USD")
    print(f"время:    {result['seconds']} c")
    print(f"трейс:    {result['trace']}")
    print("-" * 62)
    print(result["answer"] or "(ответа нет)")
    print("=" * 62)
    return 0 if result["reason"] in {"done", "done_unverified"} else 1


if __name__ == "__main__":
    sys.exit(main())
