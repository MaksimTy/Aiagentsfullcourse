.. _trace-module:

==============
Trace — Трассировка выполнения
==============

.. module:: src.agent.trace
    :synopsis: Запись и анализ хода выполнения агента

.. autoclass:: Trace
    :members:
    :undoc-members:
    :show-inheritance:

Описание модуля
================

Модуль :mod:`src.agent.trace` предоставляет класс :class:`Trace` для
автоматической записи и анализа хода выполнения агента. Все записи сохраняются
в формате JSONL (по одной записи на строку) для удобного анализа и отладки.

Назначение класса
=================

Класс :class:`Trace` служит для:

- **Аудита выполнения**: сохраняет полную историю действий агента
- **Анализа производительности**: фиксирует метрики (токены, стоимость, latency)
- **Отладки**: позволяет восстановить точку входа в каждый шаг
- **Мониторинга**: обеспечивает возможность отслеживания использования ресурсов

Архитектурные особенности
===========================

**Формат JSONL**

Все записи сохраняются в формате JSONL (JSON Lines), где каждая строка — валидный JSON.
Это позволяет:

- Легко парсить записи построчно
- Добавлять новые записи без перезаписи файла
- Использовать стандартные инструменты для анализа (jq, pandas и др.)

**Всегда включён**

Трассировка включена по умолчанию и не может быть отключена. Это обеспечивает
полную прозрачность выполнения агента.

**Автоматическое закрытие**

Файл трассировки автоматически закрывается при вызове метода :meth:`finish`,
что гарантирует сохранение всех записей.

Структура записей
==================

Записи в трассировке имеют следующие типы:

**step** — Запись шага выполнения::

    {
        "t": "step",
        "step": 1,
        "model": "gpt-4",
        "in_tokens": 150,
        "out_tokens": 50,
        "cost_usd": 0.000375,
        "latency_ms": 120,
        "window_hash": "a1b2c3d4e5f6",
        "window_msgs": 5,
        "decision": "tool_call",
        "tools": ["factorial"],
        "budget": {...}
    }

**tool** — Запись вызова инструмента::

    {
        "t": "tool",
        "step": 1,
        "tool": "factorial",
        "args": {"n": 17},
        "status": "ok",
        "error_code": null,
        "retryable": false,
        "hint": null,
        "truncated": false,
        "bytes_out": 15,
        "duration_ms": 5
    }

**finish** — Запись завершения::

    {
        "t": "finish",
        "reason": "done",
        "budget": {...},
        "answer_chars": 15
    }

Свойства класса
================

.. attribute:: run_id

    Уникальный идентификатор текущего запуска. Используется для формирования
    имени файла трассировки.

.. attribute:: path

    Путь к файлу трассировки в формате ``{runs_dir}/{run_id}.jsonl``.
    По умолчанию директория ``runs/`` в текущей рабочей директории.

Связь с другими компонентами
==============================

.. diagram::

    +------------------+         +------------------+
    |      Task        |         |      Budget      |
    +------------------+         +------------------+
           |                             |
           |                             | snapshot()
           |                             v
           |                     +------------------+
           |                     |      Trace       |
           |                     +------------------+
           |                             |
           |                             | step(), tool(), finish()
           |                             v
           |                     +------------------+
           |                     |   .jsonl file    |
           |                     +------------------+
           |                             ^
           |                             |
           v                             |
    +------------------+         +------------------+
    |       run()      |<--------+     history      |
    +------------------+         +------------------+

Пример использования
====================

Создание трассировки::

    from src.agent.trace import Trace
    
    # Создание трассировки для конкретного запуска
    trace = Trace(run_id="r-abc123")
    
    # Путь к файлу трассировки
    print(trace.path)  # runs/r-abc123.jsonl

Запись шага выполнения::

    from src.agent.trace import Trace
    from src.agent.budget import Budget
    from src.agent.llm import Reply
    
    trace = Trace(run_id="r-test")
    budget = Budget(task=task)
    
    # Создание окна диалога
    window = [
        {"role": "system", "content": "Ты помощник."},
        {"role": "user", "content": "Привет"}
    ]
    
    # Создание ответа модели
    reply = Reply(
        text="Привет! Как я могу помочь?",
        model="gpt-4",
        in_tokens=10,
        out_tokens=5,
        cost_usd=0.0001
    )
    
    # Запись шага
    trace.step(
        n=1,
        window=window,
        reply=reply,
        budget=budget,
        latency_ms=150
    )

Запись вызова инструмента::

    from src.agent.tool import ToolResult
    
    # Успешный результат
    result = ToolResult(status="ok", content="355687428096000")
    trace.tool(n=1, call={"name": "factorial", "arguments": {"n": 17}}, 
               res=result, ms=5)
    
    # Ошибка
    error_result = ToolResult(
        status="error",
        error_code="ValueError",
        hint="факториал определён только для n >= 0",
        retryable=False
    )
    trace.tool(n=2, call={"name": "factorial", "arguments": {"n": -5}},
               res=error_result, ms=2)

Запись завершения::

    trace.finish(reason="done", budget=budget, answer="Результат: 355687428096000")

Анализ трассировки
===================

Чтение трассировки из файла::

    import json
    from pathlib import Path
    
    def analyze_trace(trace_path: Path):
        """Анализирует трассировку и выводит статистику."""
        steps = []
        tools = []
        
        with open(trace_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if record['t'] == 'step':
                    steps.append(record)
                elif record['t'] == 'tool':
                    tools.append(record)
        
        print(f"Всего шагов: {len(steps)}")
        print(f"Всего вызовов инструментов: {len(tools)}")
        
        # Анализ использования токенов
        total_tokens = sum(s['in_tokens'] + s['out_tokens'] for s in steps)
        print(f"Всего токенов: {total_tokens}")
        
        # Анализ стоимости
        total_cost = sum(s['cost_usd'] for s in steps)
        print(f"Всего стоимость: ${total_cost:.6f}")

Использование с jq::

    # Фильтрация записей определённого типа
    jq -c 'select(.t == "step")' runs/r-*.jsonl
    
    # Выборка метрик
    jq -s 'map(select(.t == "step") | .cost_usd) | add' runs/r-*.jsonl

Конфигурация
=============

.. envvar:: AGENT_RUNS_DIR

    Директория для сохранения файлов трассировки.
    По умолчанию: ``runs/`` (относительно текущей рабочей директории).

Пример полного цикла выполнения::

    import hashlib
    import time
    
    from src.agent import Task, Budget, Trace
    from src.agent.budget import Budget
    from src.agent.trace import Trace
    from src.agent.llm import LLM, Reply
    from src.agent.tool import ToolRegistry
    
    def run(task: Task, tools: ToolRegistry, llm: LLM, trace: Trace) -> dict:
        budget = Budget(task=task)
        history = [{"role": "user", "content": task.prompt}]
        
        while True:
            # Проверка границ
            reason = budget.exceeded()
            if reason:
                trace.finish(reason, budget, None)
                return {"reason": reason}
            
            # Выполнение шага
            budget.steps += 1
            t0 = time.monotonic()
            reply = llm.chat(history, tools.schemas())
            latency = int((time.monotonic() - t0) * 1000)
            
            budget.tokens += reply.in_tokens + reply.out_tokens
            budget.cost_usd += reply.cost_usd
            
            # Запись шага
            trace.step(budget.steps, history, reply, budget, latency)
            
            # ... обработка ответа

Смотрите также
==============

- :class:`Budget` — класс для отслеживания ресурсов
- :class:`Reply` — структура ответа модели
- :class:`ToolResult` — результат выполнения инструмента
- :mod:`labs.lab01.starter.agent` — пример использования