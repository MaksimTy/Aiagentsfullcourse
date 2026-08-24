"""
Модуль для работы с языковыми моделями (LLM) и ответами от них.

Этот модуль предоставляет абстрактный базовый класс :class:`LLM` для
реализации адаптеров к языковым моделям, а также класс :class:`Reply`
для структурированного представления ответов модели.

Пример использования
--------------------

>>> from src.agent.llm import LLM, Reply
>>> 
>>> # Создание экземпляра ответа
>>> reply = Reply(
...     text="Привет, как я могу помочь?",
...     model="gpt-4",
...     in_tokens=100,
...     out_tokens=50,
...     cost_usd=0.001
... )
>>> 
>>> # Реализация собственного LLM-адаптера
>>> class MyLLM(LLM):
...     def chat(self, window: list[dict], tools: list[dict]) -> Reply:
...         # Логика обращения к модели
...         return Reply(text="Ответ", model="my-model")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class LLM(ABC):
    """Абстрактный базовый класс для языковых моделей (LLM).
    
    Класс определяет интерфейс для адаптеров, которые обеспечивают
    взаимодействие с языковыми моделями. Предназначен для наследования
    и реализации конкретных провайдеров (OpenAI, Anthropic, локальные модели и т.д.).
    
    Принципы дизайна:
        - Тонкий адаптер: класс LLM представляет собой лишь слой
          интеграции, а не бизнес-логику агента.
        - Единый метод :meth:`chat` для всех провайдеров.
        - Возврат структурированного объекта :class:`Reply` с метаданными.
    
    Пример реализации
    ^^^^^^^^^^^^^^^^^^
    
    .. code-block:: python
    
        from src.agent.llm import LLM, Reply
        from typing import Any
        
        class OpenAICompatLLM(LLM):
            # Адаптер для совместимых с OpenAI провайдеров
            
            def __init__(self, model: str | None = None) -> None:
                from openai import OpenAI
                self.client = OpenAI()
                self.model = model or "gpt-4"
            
            def chat(self, window: list[dict], tools: list[dict]) -> Reply:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=window,
                    tools=tools,
                    temperature=0
                )
                msg = resp.choices[0].message
                return Reply(
                    text=msg.content or "",
                    tool_calls=[
                        {"name": c.function.name, "arguments": c.function.arguments}
                        for c in (msg.tool_calls or [])
                    ],
                    model=self.model,
                    in_tokens=resp.usage.prompt_tokens,
                    out_tokens=resp.usage.completion_tokens,
                    cost_usd=self._calculate_cost(resp.usage)
                )
            
            def _calculate_cost(self, usage) -> float:
                return (usage.prompt_tokens * 0.0015 + 
                        usage.completion_tokens * 0.006) / 1000
    
    Смотрите также
    --------------
    :class:`Reply` : структура ответа от модели
    :mod:`labs.lab01.starter.agent` : пример использования в агенте
    """

    @abstractmethod
    def chat(self, window: list[dict], tools: list[dict]) -> "Reply":
        """Выполняет запрос к языковой модели.
        
        Отправляет окно диалога и список доступных инструментов в модель,
        получает ответ и возвращает его в структурированном виде.
        
        :param window: Окно диалога (контекст), представляющее собой список
            словарей с ролями (role), содержимым (content) и метаданными.
            Каждый элемент имеет вид::
            
                {"role": "system"|"user"|"assistant"|"tool", 
                 "content": str, 
                 "name": str (опционально),
                 "tool_calls": list (опционально)}
        :type window: list[dict]
        :param tools: Список схем инструментов в формате OpenAI,
            определяющих доступные функции для вызова моделью.
            Каждая схема содержит::
            
                {"type": "function",
                 "function": {
                     "name": str,
                     "description": str,
                     "parameters": dict
                 }}
        :type tools: list[dict]
        :returns: Объект :class:`Reply`, содержащий ответ модели,
            информацию о вызовах инструментов и метрики использования.
        :rtype: :class:`Reply`
        
        :raises NotImplementedError: Если метод не реализован в дочернем классе.
        
        Пример
        ^^^^^^
        
        .. code-block:: python
        
            from src.agent.llm import LLM, Reply
            
            class MyLLM(LLM):
                def chat(self, window, tools):
                    # Обращение к модели
                    response = self._call_model(window, tools)
                    return Reply(
                        text=response.text,
                        tool_calls=response.tool_calls,
                        model=self.model_name,
                        in_tokens=response.usage.prompt,
                        out_tokens=response.usage.completion,
                        cost_usd=self._calc_cost(response.usage)
                    )
        """
        pass


@dataclass
class Reply:
    """Структурированный ответ от языковой модели.
    
    Класс представляет собой dataclass для хранения ответа модели
    с метаданными о потреблении ресурсов и информации о вызовах инструментов.
    
    Атрибуты
    -------
    text : str
        Текстовое содержимое ответа модели. По умолчанию пустая строка.
        Используется для финальных ответов или промежуточных сообщений.
    tool_calls : list[dict[str, Any]]
        Список вызовов инструментов, сгенерированных моделью.
        Каждый словарь содержит::
        
            {"name": str, "arguments": dict, "id": str (опционально)}
        
        По умолчанию пустой список.
    model : str
        Идентификатор модели, использованной для генерации ответа.
        По умолчанию "fake" (для тестовых/заглушечных моделей).
    in_tokens : int
        Количество входных токенов (токены во входном окне диалога).
        Используется для расчета стоимости и мониторинга бюджета.
        По умолчанию 0.
    out_tokens : int
        Количество выходных токенов (токены в сгенерированном ответе).
        Используется для расчета стоимости и мониторинга бюджета.
        По умолчанию 0.
    cost_usd : float
        Стоимость запроса в долларах США. Вычисляется на основе токенов
        и тарифов провайдера. По умолчанию 0.0.
    
    Пример создания экземпляра
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    .. code-block:: python
    
        # Создание ответа с текстом
        reply = Reply(
            text="Результат вычисления: 355687428096000",
            model="gpt-4",
            in_tokens=150,
            out_tokens=50,
            cost_usd=0.000375
        )
        
        # Создание ответа с вызовом инструмента
        reply = Reply(
            text="",
            tool_calls=[{
                "name": "factorial",
                "arguments": {"n": 17},
                "id": "call_123"
            }],
            model="gpt-4",
            in_tokens=100,
            out_tokens=20,
            cost_usd=0.0002
        )
        
        # Пустой ответ (для тестов)
        empty_reply = Reply()
    
    Использование в системе
    ^^^^^^^^^^^^^^^^^^^^^^^
    
    Объекты :class:`Reply` используются в следующих компонентах системы:
    
    1. **Цикл агента** (:func:`labs.lab01.starter.agent.run`):
       - Проверка на пустой ответ: ``if not reply.text.strip() and not reply.tool_calls``
       - Определение типа шага: вызов инструмента или финальный ответ
       - Обновление бюджета: ``budget.tokens += reply.in_tokens + reply.out_tokens``
    
    2. **Трейсинг** (:class:`labs.lab01.starter.agent.Trace`):
       - Запись метрик: токены, стоимость, latency
       - Анализ решений модели: tool_call vs answer vs empty
    
    3. **Проверка постусловий**:
       - Извлечение текста для верификации: ``reply.text``
    
    Свойства и методы
    ^^^^^^^^^^^^^^^^^^
    
    Как dataclass, :class:`Reply` автоматически генерирует:
    
    - ``__init__`` - конструктор с параметрами по умолчанию
    - ``__repr__`` - строковое представление для отладки
    - ``__eq__`` - сравнение по значениям полей
    - ``__dataclass_fields__`` - метаданные полей
    
    Методы не определены явно, так как dataclass использует автоматическую генерацию.
    
    Смотрите также
    --------------
    :class:`LLM` : абстрактный класс, возвращающий объекты :class:`Reply`
    :class:`labs.lab01.starter.agent.ToolResult` : результат выполнения инструмента
    :class:`labs.lab01.starter.agent.Trace` : система трейсинга
    """
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = "fake"
    in_tokens: int = 0
    out_tokens: int = 0
    cost_usd: float = 0.0
