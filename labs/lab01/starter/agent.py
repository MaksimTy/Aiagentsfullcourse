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
from src.tools_repo import DEMO_TOOLS
from tools.env import load_env_file

from .llm import (
    FakeLLM, 
    OpenAICompatLLM)

load_env_file()


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
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Модуль 01: минимальное ядро агента")
    ap.add_argument("--task", default="Посчитай 17 факториал через инструмент.")
    ap.add_argument("--live", action="store_true",
                    help="использовать реального провайдера вместо FakeLLM")
    ap.add_argument("--max-steps", type=int, default=int(os.environ.get('AGENT_MAX_STEPS')))
    ap.add_argument("--max-cost", type=float, default=float(os.environ.get('AGENT_MAX_COST_USD')))
    args = ap.parse_args(argv)
    
   
    args.live = 'live' if args.live else os.environ.get('AGENT_MODE')

    run_id = "r-" + hashlib.sha256(
        (args.task + str(time.time())).encode()).hexdigest()[:6]
    trace = Trace(run_id)
    llm = {
        'live': OpenAICompatLLM(), 
        'fake':  FakeLLM() }.get(args.live)

    task = Task(prompt=args.task, 
                max_steps=args.max_steps,
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
