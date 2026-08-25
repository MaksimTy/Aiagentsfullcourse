"""
Модуль для управления бюджетами и лимитами выполнения агента.

Этот модуль предоставляет класс :class:`Budget` для отслеживания и проверки
ресурсов, используемых в процессе выполнения агента. Класс реализует принцип
"проверка до траты", предотвращая превышение лимитов.

Пример использования
--------------------

::

    from src.agent import Task, Budget
    
    # Создание задачи с лимитами
    task = Task(
        prompt="Посчитай факториал 17",
        max_steps=10,
        max_cost_usd=0.50
    )
    
    # Создание бюджета
    budget = Budget(task=task)
    
    # Проверка превышения лимитов
    reason = budget.exceeded()
    if reason:
        print(f"Бюджет исчерпан: {reason}")

Смотрите также
--------------

:class:`Task` — задача с лимитами
:class:`Trace` — трассировка выполнения
"""

import time

from dataclasses import dataclass, field

from src.agent.task import Task

# --------------------------------------------------------------------------- #
# Бюджеты
# --------------------------------------------------------------------------- #

@dataclass
class Budget:
    """
    Управление ресурсами и лимитами выполнения агента.
    
    Класс отслеживает потребление ресурсов (шаги, токены, время, стоимость)
    и проверяет, не превышены ли установленные лимиты. Реализует принцип
    "проверка до траты" — границы проверяются *до* начала нового шага.
    
    Архитектурные особенности
    --------------------------
    
    - **Автоматический старт**: таймер начинает отсчет при создании экземпляра
    - **Иммутабельная связь с задачей**: объект Task неизменяем и служит
      источником правды для лимитов
    - **Мягкие предупреждения**: приближение к 80% лимита генерирует предупреждение
    
    Атрибуты
    --------
    task : Task
        Объект задачи, определяющий лимиты выполнения. Является неизменяемым
        источником правды для проверок бюджета.
    started : float
        Временная метка начала выполнения (монотонные секунды от system.monotonic()).
        Используется для расчета фактического времени выполнения.
    steps : int
        Текущее количество выполненных шагов агента. Увеличивается при каждом
        вызове LLM.
    tokens : int
        Общее количество потребленных токенов (входных + выходных).
        Суммируется после каждого ответа модели.
    cost_usd : float
        Накопленная стоимость выполнения в долларах США. Суммируется после
        каждого ответа модели.
    empty_steps : int
        Количество последовательных пустых шагов (ответ без текста и без вызовов
        инструментов). Используется для обнаружения застревания.
    warned : bool
        Флаг, указывающий, было ли уже выдано мягкое предупреждение о превышении
        80% лимита. Предотвращает повторные предупреждения.
    
    Пример создания экземпляра
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    .. code-block:: python
    
        from src.agent import Task, Budget
        
        task = Task(
            prompt="Найди информацию о Python",
            max_steps=20,
            max_cost_usd=1.00
        )
        
        budget = Budget(task=task)
    
    Использование в цикле агента
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    .. code-block:: python
    
        def run_agent(task: Task, tools, llm, trace):
            budget = Budget(task=task)
            history = [{"role": "user", "content": task.prompt}]
            
            while True:
                # 1. Проверка границ ДО траты
                reason = budget.exceeded()
                if reason:
                    return finish(reason, history, budget, trace)
                
                # 2. Получение ответа модели
                budget.steps += 1
                reply = llm.chat(window, tools.schemas())
                budget.tokens += reply.in_tokens + reply.out_tokens
                budget.cost_usd += reply.cost_usd
                
                # 3. Запись в трассировку
                trace.step(budget.steps, window, reply, budget, latency_ms)
    
    Смотрите также
    --------------
    
    :class:`Task` — задача с лимитами
    :class:`Trace` — трассировка выполнения
    :func:`src.agent.budget.exceeded` — проверка превышения лимитов
    """
    
    task: Task
    started: float = field(default_factory=time.monotonic)
    steps: int = 0
    tokens: int = 0 
    cost_usd: float = 0.0
    empty_steps: int = 0
    warned: bool = False

    def elapsed(self) -> float:
        """
        Возвращает фактическое время выполнения с момента создания бюджета.
        
        Вычисляет разницу между текущим временем и временем начала выполнения
        с использованием монотонного таймера (time.monotonic), что обеспечивает
        корректную работу даже при изменении системного времени.
        
        :returns: Время выполнения в секундах с плавающей точкой.
        :rtype: float
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            budget = Budget(task=task)
            # ... выполнение шагов ...
            elapsed_time = budget.elapsed()
            print(f"Выполнялось {elapsed_time:.2f} секунд")
        
        Смотрите также
        --------------
        
        :attr:`task.max_seconds` — установленный лимит времени
        :meth:`exceeded` — проверка превышения лимита времени
        """
        return time.monotonic() - self.started

    def exceeded(self) -> str | None:
        """
        Проверяет, превышены ли какие-либо лимиты бюджета.
        
        Метод последовательно проверяет все лимиты (шаги, время, токены, стоимость)
        и возвращает код причины прекращения выполнения при первом найденном
        превышении. Если все лимиты соблюдены, возвращает None.
        
        Порядок проверки:
        
        1. Количество шагов (max_steps)
        2. Время выполнения (max_seconds)
        3. Количество токенов (max_tokens)
        4. Стоимость (max_cost_usd)
        
        :returns: Код причины прекращения выполнения или None, если лимиты не превышены.
        :rtype: str | None
        
        Возможные коды причин:
        
        - ``"budget_steps"`` — превышено максимальное количество шагов
        - ``"budget_time"`` — превышено максимальное время выполнения
        - ``"budget_tokens"`` — превышено максимальное количество токенов
        - ``"budget_cost"`` — превышена максимальная стоимость
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            budget = Budget(task=task)
            
            while True:
                reason = budget.exceeded()
                if reason:
                    print(f"Завершение: {reason}")
                    break
                # ... выполнение шага ...
        
        Интеграция в цикл агента
        ^^^^^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            def run(task: Task, tools, llm, trace):
                budget = Budget(task=task)
                
                while True:
                    # Критически важно: проверка ДО выполнения шага
                    reason = budget.exceeded()
                    if reason:
                        return finish(reason, history, budget, trace)
                    
                    # Выполнение шага...
        
        Смотрите также
        --------------
        
        :meth:`ratios` — отношение использования к лимитам
        :meth:`soft_warning` — мягкое предупреждение при 80% лимита
        """
        if self.steps >= self.task.max_steps:
            return "budget_steps"
        if self.elapsed() >= self.task.max_seconds:
            return "budget_time"
        if self.tokens >= self.task.max_tokens:
            return "budget_tokens"
        if self.cost_usd >= self.task.max_cost_usd:
            return "budget_cost"
        return None

    def ratios(self) -> dict[str, float]:
        """
        Возвращает отношения использованных ресурсов к установленным лимитам.
        
        Вычисляет долю использованных ресурсов относительно максимальных значений
        для каждого типа ресурса. Используется для мониторинга и генерации
        предупреждений.
        
        :returns: Словарь с отношениями для каждого типа ресурса.
        :rtype: dict[str, float]
        
        Ключи словаря:
        
        - ``"steps"`` — отношение выполненных шагов к max_steps
        - ``"time"`` — отношение времени выполнения к max_seconds
        - ``"tokens"`` — отношение токенов к max_tokens
        - ``"cost"`` — отношение стоимости к max_cost_usd
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            budget = Budget(task=task)
            # ... выполнение шагов ...
            
            ratios = budget.ratios()
            print(f"Использование: {ratios['steps']*100:.1f}% шагов")
            print(f"Стоимость: {ratios['cost']*100:.1f}% бюджета")
        
        Использование для предупреждений
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            def soft_warning(budget):
                ratios = budget.ratios()
                worst = max(ratios.values())
                if worst >= 0.8:
                    return f"Внимание: {worst*100:.0f}% лимита использовано"
                return None
        
        Смотрите также
        --------------
        
        :meth:`exceeded` — проверка превышения лимитов
        :meth:`soft_warning` — мягкое предупреждение
        """
        return {
            "steps": self.steps / max(self.task.max_steps, 1),
            "time": self.elapsed() / max(self.task.max_seconds, 1e-9),
            "tokens": self.tokens / max(self.task.max_tokens, 1),
            "cost": self.cost_usd / max(self.task.max_cost_usd, 1e-9),
        }

    def soft_warning(self) -> str | None:
        """
        Генерирует мягкое предупреждение при близком достижении лимита.
        
        Мягкий порог установлен на 80% от любого лимита. При превышении этого
        порога возвращается предупреждение, которое агент может использовать для
        раннего предупреждения о необходимости завершить работу.
        
        Метод гарантирует, что предупреждение будет выдано только один раз
        (флаг ``warned`` предотвращает повторные предупреждения).
        
        :returns: Текст предупреждения или None, если лимиты не под угрозом.
        :rtype: str | None
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            budget = Budget(task=task)
            
            # В цикле агента
            window = build_window(task, history, budget.soft_warning())
        
        Смотрите также
        --------------
        
        :meth:`ratios` — получение отношений использования
        :meth:`exceeded` — строгая проверка лимитов
        """
        """Мягкий порог 80%: агент успевает подвести итог, а не обрубается."""
        if self.warned:
            return None
        worst = max(self.ratios().values())
        if worst >= 0.8:
            self.warned = True
            return (
                "ВНИМАНИЕ: бюджет прогона израсходован более чем на 80%. "
                "Заверши работу: дай лучший доступный ответ и явно перечисли, "
                "что осталось непроверенным."
            )
        return None

    def snapshot(self) -> dict[str, str]:
        """
        Создаёт сводку текущего состояния бюджета в человеко-читаемом формате.
        
        Метод формирует словарь с строковыми представлениями текущих значений
        ресурсов и их лимитов. Используется для отображения состояния бюджета
        в интерфейсе пользователя и в логах.
        
        :returns: Словарь со сводкой состояния бюджета.
        :rtype: dict[str, str]
        
        Ключи словаря:
        
        - ``"steps"`` — количество шагов в формате "текущие/лимит"
        - ``"tokens"`` — количество токенов в формате "текущие/лимит"
        - ``"cost"`` — стоимость в формате "текущие/лимит" (с 4 знаками после запятой)
        - ``"seconds"`` — время выполнения в формате "текущие/лимит" (с 1 знаком после запятой)
        
        Пример использования
        ^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            budget = Budget(task=task)
            # ... выполнение шагов ...
            
            snapshot = budget.snapshot()
            print(f"Шаги: {snapshot['steps']}")
            print(f"Стоимость: {snapshot['cost']} USD")
        
        Использование в трассировке
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        
        .. code-block:: python
        
            # Запись состояния бюджета в трассировку
            trace.step(n, window, reply, budget, latency_ms)
            # Внутри step() вызывается budget.snapshot()
        
        Смотрите также
        --------------
        
        :meth:`ratios` — числовые отношения использования
        :attr:`steps` — текущее количество шагов
        :attr:`tokens` — текущее количество токенов
        :attr:`cost_usd` — текущая стоимость
        """
        return {
            "steps": f"{self.steps}/{self.task.max_steps}",
            "tokens": f"{self.tokens}/{self.task.max_tokens}",
            "cost": f"{self.cost_usd:.4f}/{self.task.max_cost_usd}",
            "seconds": f"{self.elapsed():.1f}/{self.task.max_seconds}",
        }
