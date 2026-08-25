"""
Модуль ядра агента.

Этот модуль предоставляет основные компоненты для создания и выполнения
инженерных агентов. Включает классы для управления задачами, бюджетами,
трассировкой выполнения, инструментами и взаимодействием с языковыми моделями.

Основные компоненты
===================

Классы задач и бюджета
-----------------------

- :class:`Task` — Описание задачи с лимитами и критериями успеха
- :class:`Budget` — Управление ресурсами и проверка лимитов выполнения

Инструменты
-----------

- :class:`Tool` — Описание инструмента (функции) для вызова агентом
- :class:`ToolResult` — Результат выполнения инструмента
- :class:`ToolRegistry` — Реестр и менеджер инструментов

Взаимодействие с LLM
--------------------

- :class:`LLM` — Абстрактный базовый класс для языковых моделей
- :class:`Reply` — Структурированный ответ от модели

Трассировка
-----------

- :class:`Trace` — Запись и анализ хода выполнения агента

Пример использования
====================

Создание простого агента::

    from src.agent import Task, Budget, Trace, Tool, ToolRegistry, LLM, Reply
    
    # Определение инструмента
    def calculator(expression: str) -> str:
        return str(eval(expression))
    
    tool = Tool(
        name="calculator",
        description="Вычисляет математическое выражение",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "математическое выражение"}
            },
            "required": ["expression"]
        },
        fn=calculator
    )
    
    # Создание реестра инструментов
    tools = ToolRegistry([tool])
    
    # Определение задачи
    task = Task(
        prompt="Посчитай 17 * 23",
        max_steps=10,
        max_cost_usd=0.50
    )
    
    # Создание трассировки
    trace = Trace(run_id="run-001")
    
    # Использование...

Смотрите также
==============

- :mod:`labs.lab01.starter.agent` — Пример полного цикла выполнения агента
- :mod:`src.agent.budget` — Документация по классу Budget
- :mod:`src.agent.task` — Документация по классу Task
- :mod:`src.agent.trace` — Документация по классу Trace
"""

from .task import Task
from .budget import Budget
from .tool import (
    Tool, 
    ToolResult, 
    ToolRegistry)
from .trace import Trace
from .llm import (
    LLM, 
    Reply)



__all__ = [
    "Task",
    "Budget",
    "Tool", 
    "ToolResult", 
    "ToolRegistry",
    "Trace",
    "LLM", 
    "Reply"
]