"""
Модуль для трассировки и анализа выполнения агента.

Этот модуль предоставляет класс :class:`Trace` для автоматической записи
и анализа хода выполнения агента. Все записи сохраняются в формате JSONL
(по одной записи на строку) для удобного анализа и отладки.

Пример использования
--------------------

::

    from src.agent import Task, Budget, Trace
    
    # Создание трассировки
    trace = Trace(run_id="run-001")
    
    # Запись шага
    trace.step(n=1, window=window, reply=reply, budget=budget, latency_ms=150)
    
    # Запись вызова инструмента
    trace.tool(n=1, call=call, res=result, ms=5)
    
    # Запись завершения
    trace.finish(reason="done", budget=budget, answer="Результат: 355687428096000")

Смотрите также
--------------

:class:`Budget` — класс для отслеживания ресурсов
:class:`Reply` — структура ответа модели
:class:`ToolResult` — результат выполнения инструмента
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Any

from src.agent.llm import Reply
from src.agent.tool import ToolResult
from src.agent.budget import Budget



# --------------------------------------------------------------------------- #
# Трейс: JSONL, всегда включён
# --------------------------------------------------------------------------- #


class Trace:
    """
    Трассировка и анализ хода выполнения агента.
    
    Класс обеспечивает автоматическую запись полной истории выполнения агента
    в формате JSONL (JSON Lines). Все записи сохраняются немедленно после
    каждого события, что обеспечивает надёжность даже при падении процесса.
    
    Архитектурные особенности
    --------------------------
    
    - **Всегда включён**: трассировка никогда не отключается
    - **Формат JSONL**: каждая строка — валидный JSON, удобен для анализа
    - **Автоматическое закрытие**: файл закрывается при вызове :meth:`finish`
    - **Немедленное сохранение**: каждая запись сразу записывается на диск
    
    Атрибуты
    --------
    run_id : str
        Уникальный идентификатор текущего запуска. Используется для формирования
        имени файла трассировки в формате ``{run_id}.jsonl``.
    path : Path
        Полный путь к файлу трассировки. Формируется как
        ``{AGENT_RUNS_DIR}/{run_id}.jsonl``, где ``AGENT_RUNS_DIR`` по умолчанию
        равен ``runs/``.
    
    Пример создания экземпляра
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    .. code-block:: python
    
        from src.agent import Trace
        
        # Создание трассировки с указанным ID
        trace = Trace(run_id="r-abc123")
        print(trace.path)  # runs/r-abc123.jsonl
    
    Использование в цикле агента
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    .. code-block:: python
    
        from src.agent import Task, Budget, Trace
        
        def run(task: Task, tools, llm, trace: Trace) -> dict:
            budget = Budget(task=task)
            history = [{"role": "user", "content": task.prompt}]
            
            while True:
                reason = budget.exceeded()
                if reason:
                    trace.finish(reason, budget, None)
                    return {"reason": reason}
                
                # Выполнение шага...
                budget.steps += 1
                reply = llm.chat(window, tools.schemas())
                
                # Запись в трассировку
                trace.step(budget.steps, window, reply, budget, latency_ms)
    
    Смотрите также
    --------------
    
    :class:`Budget` — класс для отслеживания ресурсов
    :class:`Reply` — структура ответа модели
    :class:`ToolResult` — результат выполнения инструмента
    """
    
    def __init__(self, run_id: str) -> None:
        """
        Инициализирует трассировку для указанного запуска.
        
        Создаёт директорию для трассировок (если не существует) и открывает
        файл для записи. Файл открывается в режиме добавления (append), что
        позволяет добавлять записи без перезаписи существующих.
        
        :param run_id: Уникальный идентификатор запуска. Используется для
            формирования имени файла трассировки.
        :type run_id: str
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            trace = Trace(run_id="r-20240115-001")
            print(trace.path)  # runs/r-20240115-001.jsonl
        
        Конфигурация
        ^^^^^^^^^^^^
        
        Директория для трассировок задаётся переменной окружения
        ``AGENT_RUNS_DIR``. По умолчанию используется ``runs/``.
        
        .. code-block:: python
        
            import os
            os.environ["AGENT_RUNS_DIR"] = "/var/log/agent/traces"
            trace = Trace(run_id="r-001")  # Файл будет в /var/log/agent/traces/
        """
        RUNS_DIR = Path(os.environ.get("AGENT_RUNS_DIR", "runs"))
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.path = RUNS_DIR / f"{run_id}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def _write(self, rec: dict[str, Any]) -> None:
        """
        Записывает запись в файл трассировки.
        
        Метод добавляет общие поля (run_id, ts) к записи и записывает её
        в файл в формате JSON. Каждая запись записывается на отдельную строку
        с немедленным сбросом буфера (flush).
        
        :param rec: Словарь с данными записи.
        :type rec: dict[str, Any]
        
        Примечания
        ----------
        
        - Запись сразу сбрасывается на диск (flush)
        - Дата и время записываются в формате ISO 8601 (UTC)
        - JSON кодируется с ensure_ascii=False для корректной работы с русским текстом
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            self._write({
                "t": "step",
                "step": 1,
                "model": "gpt-4",
                "cost_usd": 0.000375
            })
        """
        rec = {"run_id": self.run_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                          time.gmtime()), **rec}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    @staticmethod
    def _hash(obj: Any) -> str:
        """
        Вычисляет хеш объекта для идентификации диалогового окна.
        
        Метод сериализует объект в JSON со сортировкой ключей и вычисляет
        SHA-256 хеш, возвращая первые 12 символов. Используется для
        обнаружения повторяющихся окон диалога.
        
        :param obj: Объект для хеширования (обычно список сообщений).
        :type obj: Any
        :returns: 12-символьный хеш в виде строки.
        :rtype: str
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            window1 = [{"role": "user", "content": "Привет"}]
            window2 = [{"role": "user", "content": "Привет"}]
            
            h1 = Trace._hash(window1)
            h2 = Trace._hash(window2)
            print(h1 == h2)  # True
        
        Примечания
        ----------
        
        - Хеш вычисляется от нормализованных данных (сортировка ключей)
        - Используется для анализа повторяющихся запросов
        - Длина 12 символов обеспечивает достаточную уникальность
        """
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def step(self, n: int, window: list[dict], reply: "Reply", budget: Budget,
             latency_ms: int) -> None:
        """
        Записывает запись о шаге выполнения агента.
        
        Метод фиксирует полную информацию о шаге: входящее окно диалога,
        ответ модели, потребление ресурсов и время выполнения.
        
        :param n: Номер шага (номерация с 1).
        :type n: int
        :param window: Окно диалога, отправленное в модель.
        :type window: list[dict]
        :param reply: Ответ от языковой модели.
        :type reply: Reply
        :param budget: Текущее состояние бюджета.
        :type budget: Budget
        :param latency_ms: Время выполнения шага в миллисекундах.
        :type latency_ms: int
        
        Формат записи
        -------------
        
        Запись содержит следующие поля:
        
        - ``t`` — тип записи ("step")
        - ``step`` — номер шага
        - ``model`` — идентификатор модели
        - ``in_tokens``, ``out_tokens`` — количество токенов
        - ``cost_usd`` — стоимость шага
        - ``latency_ms`` — задержка выполнения
        - ``window_hash`` — хеш окна диалога
        - ``window_msgs`` — количество сообщений в окне
        - ``decision`` — тип решения: "tool_call", "answer" или "empty"
        - ``tools`` — список вызванных инструментов
        - ``budget`` — сводка состояния бюджета
        
        Пример записи
        ^^^^^^^^^^^^^
        
        .. code-block:: json
        
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
                "budget": {
                    "steps": "1/10",
                    "tokens": "200/200000",
                    "cost": "0.0004/0.5000",
                    "seconds": "0.1/300.0"
                }
            }
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            from src.agent import Budget, Trace
            from src.agent.llm import Reply
            
            trace = Trace(run_id="r-test")
            budget = Budget(task=task)
            reply = Reply(text="", tool_calls=[{"name": "factorial", "arguments": {"n": 17}}])
            
            trace.step(
                n=1,
                window=[{"role": "user", "content": "Посчитай 17!"}],
                reply=reply,
                budget=budget,
                latency_ms=150
            )
        """
        self._write({
            "t": "step", 
            "step": n, 
            "model": reply.model,
            "in_tokens": reply.in_tokens, 
            "out_tokens": reply.out_tokens,
            "cost_usd": round(reply.cost_usd, 6), 
            "latency_ms": latency_ms,
            "window_hash": self._hash(window), 
            "window_msgs": len(window),
            "decision": "tool_call" if reply.tool_calls else
                        ("answer" if reply.text.strip() else "empty"),
            "tools": [c["name"] for c in reply.tool_calls],
            "budget": budget.snapshot(),
        })

    def tool(self, n: int, call: dict, res: ToolResult, ms: int) -> None:
        """
        Записывает запись о вызове инструмента.
        
        Метод фиксирует информацию об вызове инструмента: имя, аргументы,
        результат, статус и время выполнения.
        
        :param n: Номер шага, в котором был выполнен вызов.
        :type n: int
        :param call: Словарь с данными о вызове инструмента.
            Ожидается формат: ``{"name": str, "arguments": dict}``.
        :type call: dict
        :param res: Результат выполнения инструмента.
        :type res: ToolResult
        :param ms: Время выполнения инструмента в миллисекундах.
        :type ms: int
        
        Формат записи
        -------------
        
        Запись содержит следующие поля:
        
        - ``t`` — тип записи ("tool")
        - ``step`` — номер шага
        - ``tool`` — имя инструмента
        - ``args`` — аргументы вызова
        - ``status`` — статус: "ok" или "error"
        - ``error_code`` — код ошибки (если статус "error")
        - ``retryable`` — признак повторного вызова
        - ``hint`` — подсказка для исправления ошибки
        - ``truncated`` — признак обрезки вывода
        - ``bytes_out`` — размер вывода в байтах
        - ``duration_ms`` — длительность выполнения
        
        Пример записи успешного вызова
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: json
        
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
        
        Пример записи ошибки
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: json
        
            {
                "t": "tool",
                "step": 2,
                "tool": "factorial",
                "args": {"n": -5},
                "status": "error",
                "error_code": "ValueError",
                "retryable": false,
                "hint": "факториал определён только для n >= 0",
                "truncated": false,
                "bytes_out": 0,
                "duration_ms": 2
            }
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            from src.agent import Trace
            from src.agent.tool import ToolResult
            
            trace = Trace(run_id="r-test")
            
            # Успешный вызов
            result = ToolResult(status="ok", content="355687428096000")
            trace.tool(
                n=1,
                call={"name": "factorial", "arguments": {"n": 17}},
                res=result,
                ms=5
            )
            
            # Ошибочный вызов
            error_result = ToolResult(
                status="error",
                error_code="ValueError",
                hint="факториал определён только для n >= 0",
                retryable=False
            )
            trace.tool(
                n=2,
                call={"name": "factorial", "arguments": {"n": -5}},
                res=error_result,
                ms=2
            )
        """
        self._write({
            "t": "tool", 
            "step": n, 
            "tool": call["name"],
            "args": call.get("arguments", {}), 
            "status": res.status,
            "error_code": res.error_code, 
            "retryable": res.retryable,
            "hint": res.hint, 
            "truncated": res.truncated,
            "bytes_out": len(res.content), 
            "duration_ms": ms,
        })

    def finish(self, reason: str, budget: Budget, answer: str | None) -> None:
        """
        Записывает запись о завершении выполнения агента.
        
        Метод фиксирует причину завершения, финальное состояние бюджета
        и длину ответа. После вызова метода файл трассировки закрывается.
        
        :param reason: Причина завершения выполнения.
            Возможные значения:
            
            - ``"done"`` — задача успешно выполнена
            - ``"done_unverified"`` — задача выполнена, но не проверена постусловием
            - ``"budget_steps"`` — превышен лимит шагов
            - ``"budget_time"`` — превышен лимит времени
            - ``"budget_tokens"`` — превышен лимит токенов
            - ``"budget_cost"`` — превышена лимит стоимости
            - ``"no_progress"`` — отсутствует прогресс (пустые шаги)
            - ``"repeated_action"`` — повторяющийся вызов инструмента
        :type reason: str
        :param budget: Финальное состояние бюджета.
        :type budget: Budget
        :param answer: Финальный ответ агента (может быть None).
        :type answer: str | None
        
        Формат записи
        -------------
        
        Запись содержит следующие поля:
        
        - ``t`` — тип записи ("finish")
        - ``reason`` — причина завершения
        - ``budget`` — сводка финального состояния бюджета
        - ``answer_chars`` — количество символов в ответе
        
        Пример записи
        ^^^^^^^^^^^^^
        
        .. code-block:: json
        
            {
                "t": "finish",
                "reason": "done",
                "budget": {
                    "steps": "3/10",
                    "tokens": "500/200000",
                    "cost": "0.0025/0.5000",
                    "seconds": "1.5/300.0"
                },
                "answer_chars": 15
            }
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            from src.agent import Budget, Trace
            
            trace = Trace(run_id="r-test")
            budget = Budget(task=task)
            
            # Завершение с успешным результатом
            trace.finish(
                reason="done",
                budget=budget,
                answer="Результат: 355687428096000"
            )
            
            # Файл трассировки закрыт, запись сохранена
        """
        self._write({"t": "finish", "reason": reason,
                     "budget": budget.snapshot(),
                     "answer_chars": len(answer or "")})
        self._fh.close()