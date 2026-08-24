"""
Модуль для работы с инструментами (tools) в агенте.

Этот модуль предоставляет базовые классы для описания, выполнения и управления
инструментами в системе агентов. Инструменты позволяют агенту взаимодействовать
с внешними системами, выполнять вычисления, читать файлы и многое другое.

Принцип работы:
    - Инструменты описываются классом :class:`Tool`
    - Результат выполнения инструмента представлен классом :class:`ToolResult`
    - :class:`ToolRegistry` управляет коллекцией инструментов и их выполнением

Пример использования::

    from src.agent.tool import Tool, ToolRegistry

    # Определение инструмента
    def get_weather(city: str) -> str:
        return f"Погода в {city}: солнечно"

    tool = Tool(
        name="get_weather",
        description="Получить погоду в указанном городе",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "название города"}
            },
            "required": ["city"]
        },
        fn=get_weather
    )

    # Создание реестра инструментов
    registry = ToolRegistry([tool])

    # Получение схемы для передачи в модель
    schemas = registry.schemas()

    # Вызов инструмента
    result = registry.invoke({"name": "get_weather", "arguments": {"city": "Москва"}})
"""

import os
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable


# --------------------------------------------------------------------------- #
# Инструменты: ошибка возвращается как данные
# --------------------------------------------------------------------------- #


@dataclass
class Tool:
    """
    Описание инструмента (tool) для агента.

    Класс представляет собой dataclass, описывающий функциональность инструмента,
    который может быть вызван агентом. Инструменты позволяют агенту выполнять
    действия во внешнем мире, такие как вычисления, чтение файлов, взаимодействие
    с API и другие операции.

    Агент использует экземпляры этого класса для:

    - Формирования подсказок для LLM о доступных инструментах
    - Валидации аргументов перед выполнением
    - Выполнения функциональности через метод :meth:`ToolRegistry.invoke`

    Атрибуты
    --------
    name : str
        Уникальное имя инструмента, используемое для идентификации при вызове.
        Должно быть корректным идентификатором (без пробелов, спецсимволов).
        Используется как ключ в словаре :attr:`ToolRegistry._tools`.
    description : str
        Человеко-читаемое описание назначения инструмента.
        Используется для генерации подсказок модели и документации.
        Должно быть информативным и понятным для конечного пользователя.
    parameters : dict[str, Any]
        Схема параметров инструмента в формате JSON Schema.
        Определяет типы, описания и обязательность аргументов.

        Ожидается структура::

            {
                "type": "object",
                "properties": {
                    "param_name": {
                        "type": "string|integer|number|boolean|array|object",
                        "description": "Описание параметра"
                    }
                },
                "required": ["param_name1", "param_name2"]
            }

        Полный спектр JSON Schema поддерживается, включая:
        - Типы: string, integer, number, boolean, array, object
        - Ограничения: minimum, maximum, enum, pattern, format
        - Вложенные структуры для сложных типов

    fn : Callable[..., str]
        Функция-исполнитель, которая принимает аргументы из ``parameters``
        и возвращает строку с результатом выполнения.

        Сигнатура функции должна соответствовать схеме параметров.
        Аргументы передаются по имени через **kwargs.

    Пример
    ------
    Определение простого инструмента для вычисления факториала::

        import math

        def _factorial(n: int) -> str:
            if n < 0:
                raise ValueError("факториал определён только для n >= 0")
            if n > 2000:
                raise ValueError("n слишком велико, предел 2000")
            return str(math.factorial(n))

        factorial_tool = Tool(
            name="factorial",
            description=(
                "Точно вычисляет факториал целого неотрицательного n. "
                "Возвращает десятичную запись числа. Предел n = 2000."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2000,
                        "description": "неотрицательное целое"
                    }
                },
                "required": ["n"]
            },
            fn=_factorial
        )

    Примечания
    ----------
    - Все инструменты должны быть синхронными функциями
    - Функция должна возвращать строку (str), даже если результат другого типа
    - Исключения внутри функции перехватываются и преобразуются в ToolResult
      с кодом ошибки и подсказкой
    - Инструменты не имеют доступа к внешним состояниям (это обеспечивает
      предсказуемость и тестируемость)

    Смотрите также
    --------------
    :class:`ToolResult` : результат выполнения инструмента
    :class:`ToolRegistry` : реестр и менеджер инструментов
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]


@dataclass
class ToolResult:
    """
    Результат выполнения инструмента.

    Класс представляет собой dataclass для инкапсуляции результата выполнения
    инструмента. Ключевой принцип: **ошибки инструмента - это данные, а не исключения**.
    Это позволяет агенту обрабатывать ошибки как обычный вывод, не перехватывая
    исключения на верхних уровнях.

    Атрибуты
    --------
    status : str
        Статус выполнения: ``"ok"`` или ``"error"``.

        - ``"ok"`` — инструмент успешно выполнен
        - ``"error"`` — произошла ошибка при выполнении
    content : str, optional
        Содержимое результата выполнения. Для успешных выполнений
        (status="ok") содержит текстовый результат. По умолчанию пустая строка.
    error_code : str | None, optional
        Код ошибки для диагностики. Используется только при status="error".
        Примеры кодов ошибок:

        - ``"unknown_tool"`` — инструмент не найден в реестре
        - ``"missing_argument"`` — не переданы обязательные аргументы
        - ``"bad_arguments"`` — неверные типы или значения аргументов
        - Имя исключения Python (например, ``"ValueError"``, ``"TypeError"``)

        По умолчанию ``None``.
    hint : str | None, optional
        Подсказка для исправления ошибки или понимания результата.
        Используется для обогащения сообщения об ошибке.
        По умолчанию ``None``.
    retryable : bool, optional
        Флаг, указывающий, можно ли повторить выполнение с теми же аргументами.

        - ``True`` — повторный вызов может помочь (например, временная ошибка)
        - ``False`` — повторный вызов с теми же аргументами не поможет

        По умолчанию ``False``.
    truncated : bool, optional
        Флаг, указывающий, был ли обрезан вывод инструмента.
        Обрезка происходит при превышении лимита символов
        (переменная окружения ``TOOL_OUTPUT_MAX_CHARS``).
        По умолчанию ``False``.

    Методы
    --------
    to_model() -> str
        Формирует строку, которую увидит LLM при получении результата.
        Для успешных результатов — содержимое с пометкой об обрезке.
        Для ошибок — сформированное сообщение об ошибке с подсказкой.

    Пример использования
    -------------------
    Получение результата от инструмента::

        from src.agent.tool import ToolRegistry

        registry = ToolRegistry([factorial_tool])

        # Успешное выполнение
        result = registry.invoke({"name": "factorial", "arguments": {"n": 5}})
        if result.status == "ok":
            print(f"Результат: {result.content}")  # "120"

        # Обработка ошибки
        result = registry.invoke({"name": "factorial", "arguments": {"n": -1}})
        if result.status == "error":
            print(f"Ошибка: {result.error_code}")  # "ValueError"
            print(f"Подсказка: {result.hint}")

    Взаимодействие с LLM
    -------------------
    Метод :meth:`to_model` формирует строку для передачи в историю диалога::

        # Успешный результат
        result = ToolResult(status="ok", content="120")
        model_output = result.to_model()  # "120"

        # Ошибка с подсказкой
        result = ToolResult(
            status="error",
            error_code="missing_argument",
            hint="Не переданы обязательные аргументы: n",
            retryable=True
        )
        model_output = result.to_model()
        # "ОШИБКА missing_argument Подсказка: Не переданы обязательные аргументы: n
        #  Повтор возможен."

    Примечания
    ----------
    - Все атрибуты, кроме ``status``, имеют значения по умолчанию
    - При успешном выполнении (status="ok") поле ``content`` всегда заполняется
    - При ошибке (status="error") заполняются ``error_code``, ``hint``, ``retryable``
    - Обрезка вывода происходит до создания ToolResult, а флаг truncated
      сообщает об этом вызывающему коду

    Смотрите также
    --------------
    :class:`Tool` — описание инструмента
    :class:`ToolRegistry` — менеджер инструментов
    """

    status: str                 # "ok" | "error"
    content: str = ""
    error_code: str | None = None
    hint: str | None = None
    retryable: bool = False
    truncated: bool = False

    def to_model(self) -> str:
        """
        Формирует строку, которую увидит LLM при получении результата.

        Метод создает человеко-читаемое представление результата для передачи
        в окно диалога (window) и историю (history). Для ошибок сообщение
        должно быть "действенным" — позволять агенту понять, что пошло не так,
        и как это исправить.

        Возвращаемое значение
        ---------------------
        str
            Текст результата для LLM:

            - При status="ok": содержимое ``content`` с пометкой об обрезке,
              если ``truncated=True``
            - При status="error": сформированное сообщение об ошибке вида::

                ОШИБКА {error_code} Подсказка: {hint} Повтор {может быть/не поможет}

        Примеры
        -------
        Успешный результат без обрезки::

            >>> result = ToolResult(status="ok", content="Результат: 120")
            >>> result.to_model()
            'Результат: 120'

        Успешный результат с обрезкой::

            >>> result = ToolResult(status="ok", content="Очень длинный вывод...", truncated=True)
            >>> result.to_model()
            'Очень длинный вывод...\\n[...вывод обрезан...]'

        Ошибка с подсказкой::

            >>> result = ToolResult(
            ...     status="error",
            ...     error_code="missing_argument",
            ...     hint="Не переданы обязательные аргументы: n",
            ...     retryable=True
            ... )
            >>> result.to_model()
            'ОШИБКА missing_argument Подсказка: Не переданы обязательные аргументы: n Повтор возможен.'

        Ошибка без подсказки::

            >>> result = ToolResult(
            ...     status="error",
            ...     error_code="unknown_tool",
            ...     retryable=False
            ... )
            >>> result.to_model()
            'ОШИБКА unknown_tool Повтор с теми же аргументами не поможет.'

        Примечания
        ----------
        - Формат сообщения об ошибке на русском языке
        - Если ``hint`` не задан, подсказка не включается в вывод
        - Фраза "Повтор возможен" или "Повтор с теми же аргументами не поможет"
          добавляется автоматически в зависимости от ``retryable``
        """
        if self.status == "ok":
            tail = "\n[...вывод обрезан...]" if self.truncated else ""
            return self.content + tail
        parts = [f"ОШИБКА {self.error_code}"]
        if self.hint:
            parts.append(f"Подсказка: {self.hint}")
        parts.append("Повтор с теми же аргументами не поможет."
                     if not self.retryable else "Повтор возможен.")
        return " ".join(parts)


class ToolRegistry:
    """
    Реестр и менеджер инструментов.

    Класс управляет коллекцией инструментов и обеспечивает их выполнение.
    Служит как центральный пункт для:

    - Хранения и поиска инструментов по имени
    - Генерации схем для передачи в LLM
    - Выполнения инструментов с обработкой ошибок
    - Обнаружения повторяющихся вызовов

    Архитектурные особенности
    -------------------------
    - Все ошибки инструментов преобразуются в :class:`ToolResult`, а не
      выбрасываются как исключения
    - Выполнение инструментов синхронное
    - Реестр не сохраняет состояние между вызовами (stateless)

    Атрибуты
    --------
    _tools : dict[str, Tool]
        Внутренний словарь инструментов, сопоставляющий имя инструмента
        с его описанием. Создается при инициализации.

    Методы
    --------
    schemas() -> list[dict[str, Any]]
        Генерирует схемы инструментов в формате OpenAI Functions API.
    fingerprint(call: dict[str, Any]) -> str
        Вычисляет хеш вызова для обнаружения повторов.
    invoke(call: dict[str, Any]) -> ToolResult
        Выполняет инструмент и возвращает результат.

    Пример использования
    -------------------
    Создание и использование реестра::

        from src.agent.tool import Tool, ToolRegistry

        # Определение инструментов
        def add(a: int, b: int) -> str:
            return str(a + b)

        add_tool = Tool(
            name="add",
            description="Складывает два числа",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "первое число"},
                    "b": {"type": "integer", "description": "второе число"}
                },
                "required": ["a", "b"]
            },
            fn=add
        )

        # Создание реестра
        registry = ToolRegistry([add_tool])

        # Получение схем для LLM
        schemas = registry.schemas()
        # [{'type': 'function', 'function': {...}}]

        # Вызов инструмента
        result = registry.invoke({"name": "add", "arguments": {"a": 5, "b": 3}})
        print(result.status)    # "ok"
        print(result.content)   # "8"

    Методы
    --------
    """

    def __init__(self, tools: Iterable[Tool]) -> None:
        """
        Инициализирует реестр инструментов.

        Параметры
        ---------
        tools : Iterable[Tool]
            Итерируемый объект, содержащий инструменты для добавления в реестр.
            Каждый инструмент должен иметь уникальное имя.

        Пример
        ------
        .. code-block:: python

            from src.agent.tool import Tool, ToolRegistry

            tool1 = Tool(name="tool1", description="...", parameters={}, fn=lambda: "1")
            tool2 = Tool(name="tool2", description="...", parameters={}, fn=lambda: "2")

            registry = ToolRegistry([tool1, tool2])
        """
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        """
        Генерирует схемы инструментов в формате OpenAI Functions API.

        Метод преобразует внутреннее представление инструментов в формат,
        который понимает LLM (OpenAI Functions API). Используется для передачи
        списка доступных инструментов в вызов модели.

        Возвращаемое значение
        ---------------------
        list[dict[str, Any]]
            Список словарей, каждый из которых содержит схему одного инструмента::

                [
                    {
                        "type": "function",
                        "function": {
                            "name": "tool_name",
                            "description": "описание",
                            "parameters": {...}
                        }
                    },
                    ...
                ]

        Пример использования
        ------------------
        .. code-block:: python

            registry = ToolRegistry([factorial_tool, read_file_tool])
            schemas = registry.schemas()

            # schemas можно передать в LLM
            response = llm.chat(window, schemas)

        Смотрите также
        --------------
        :meth:`invoke` — выполнение инструмента
        :meth:`fingerprint` — вычисление хеша вызова
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    @staticmethod
    def fingerprint(call: dict[str, Any]) -> str:
        """
        Вычисляет хеш (имя + нормализованные аргументы) для детектора повторов.

        Метод создает уникальный идентификатор вызова инструмента на основе
        имени инструмента и аргументов. Используется для обнаружения повторяющихся
        вызовов, чтобы предотвратить бесконечные циклы агента.

        Параметры
        ---------
        call : dict[str, Any]
            Словарь с данными о вызове инструмента. Ожидается структура::

                {
                    "name": "название_инструмента",
                    "arguments": {"param": "value", ...}
                }

        Возвращаемое значение
        ---------------------
        str
            12-символьный хеш в виде строки, полученный с помощью SHA-256.

        Пример
        ------
        .. code-block:: python

            call1 = {"name": "factorial", "arguments": {"n": 5}}
            call2 = {"name": "factorial", "arguments": {"n": 5}}
            call3 = {"name": "factorial", "arguments": {"n": 6}}

            fp1 = ToolRegistry.fingerprint(call1)
            fp2 = ToolRegistry.fingerprint(call2)
            fp3 = ToolRegistry.fingerprint(call3)

            print(fp1 == fp2)  # True - одинаковые вызовы
            print(fp1 == fp3)  # False - разные аргументы

        Примечания
        ----------
        - Хеш вычисляется от нормализованных аргументов (сортировка ключей)
        - Используется для предотвращения бесконечных повторов в цикле агента
        - Длина хеша 12 символов обеспечивает достаточную уникальность
          при минимальном размере для хранения

        Смотрите также
        --------------
        :meth:`invoke` — выполнение инструмента
        """
        payload = json.dumps(
            {"n": call["name"], "a": call.get("arguments", {})},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def invoke(self, call: dict[str, Any]) -> ToolResult:
        """
        Выполняет инструмент и возвращает результат.

        Метод принимает запрос на выполнение инструмента, валидирует его,
        вызывает соответствующую функцию и возвращает результат. Все ошибки
        преобразуются в :class:`ToolResult` с соответствующим кодом ошибки.

        Параметры
        ---------
        call : dict[str, Any]
            Словарь с данными о вызове инструмента. Ожидается структура::

                {
                    "name": "название_инструмента",
                    "arguments": {"param": "value", ...}  # опционально
                }

        Возвращаемое значение
        ---------------------
        ToolResult
            Результат выполнения инструмента:

            - При успехе: ``ToolResult(status="ok", content="результат")``
            - При ошибке: ``ToolResult(status="error", error_code="...", hint="...")``

        Примеры использования
        -------------------
        Успешное выполнение::

            >>> result = registry.invoke({"name": "factorial", "arguments": {"n": 5}})
            >>> result.status
            'ok'
            >>> result.content
            '120'

        Ошибка: инструмент не найден::

            >>> result = registry.invoke({"name": "unknown", "arguments": {}})
            >>> result.status
            'error'
            >>> result.error_code
            'unknown_tool'
            >>> result.hint
            'Доступные инструменты: factorial, read_file'

        Ошибка: недостающие аргументы::

            >>> result = registry.invoke({"name": "factorial", "arguments": {}})
            >>> result.status
            'error'
            >>> result.error_code
            'missing_argument'
            >>> result.hint
            'Не переданы обязательные аргументы: n'

        Ошибка: неверные аргументы::

            >>> result = registry.invoke({"name": "factorial", "arguments": {"n": -5}})
            >>> result.status
            'error'
            >>> result.error_code
            'ValueError'

        Логика обработки ошибок
        -----------------------
        Метод последовательно проверяет:

        1. **Существование инструмента**: если инструмент не найден,
           возвращается ошибка ``unknown_tool`` с подсказкой о доступных инструментах

        2. **Обязательные аргументы**: проверяется, что все обязательные
           параметры переданы. Если нет — возвращается ошибка ``missing_argument``

        3. **Выполнение функции**: функция вызывается с переданными аргументами.
           Исключения перехватываются:

           - ``TypeError`` → ошибка ``bad_arguments`` (неверные типы/аргументы)
           - Любые другие исключения → ошибка с именем исключения

        4. **Обрезка вывода**: если длина результата превышает лимит
           (``TOOL_OUTPUT_MAX_CHARS``), вывод обрезается и устанавливается
           флаг ``truncated=True``

        Примечания
        ----------
        - Метод **никогда не выбрасывает исключения наружу** — всё превращается в ToolResult
        - Это позволяет агенту обрабатывать ошибки как обычные данные
        - Выполнение синхронное, без поддержки асинхронных функций
        - Результат всегда строки (преобразуется через str())

        Смотрите также
        --------------
        :meth:`schemas` — получение схем инструментов
        :meth:`fingerprint` — вычисление хеша вызова
        """
        name = call.get("name", "")
        args = call.get("arguments", {}) or {}
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                status="error", error_code="unknown_tool", retryable=False,
                hint="Доступные инструменты: " + ", ".join(sorted(self._tools)),
            )
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        if missing:
            return ToolResult(
                status="error", error_code="missing_argument", retryable=True,
                hint=f"Не переданы обязательные аргументы: {', '.join(missing)}",
            )
        try:
            out = str(tool.fn(**args))
        except TypeError as exc:
            return ToolResult(status="error", error_code="bad_arguments",
                              retryable=True, hint=str(exc))
        except Exception as exc:                      # noqa: BLE001
            return ToolResult(status="error", error_code=type(exc).__name__,
                              retryable=False, hint=str(exc)[:300])
        truncated = len(out) > int(os.environ.get("TOOL_OUTPUT_MAX_CHARS", 1000000))
        return ToolResult(status="ok", content=out[:int(os.environ.get("TOOL_OUTPUT_MAX_CHARS", 1000000))],
                          truncated=truncated)

