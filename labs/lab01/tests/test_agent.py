"""
Тесты для модуля 01: Основы агентов.

Проверяет выполнение первых трёх пунктов чек-листа приёмки:
1. Цикл не может выполниться больше max_steps раз
2. Реализованы все четыре бюджета, и каждый имеет тест на срабатывание
3. Есть мягкий порог 80%, и предупреждение действительно попадает в контекст
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent import Budget, Task, Trace, Tool, ToolRegistry, Reply, LLM

from src.tools_repo import DEMO_TOOLS

from labs.lab01.starter.llm import (FakeLLM, OpenAICompatLLM, InfiniteToolCallLLM)



# --------------------------------------------------------------------------- #
# Тест 1: Цикл не может выполниться больше max_steps раз
# --------------------------------------------------------------------------- #

def test_cycle_respects_max_steps():
    """
    Тест: цикл не может выполниться больше max_steps раз.
    
    Проверяется, что при фиктивной модели, которая всегда просит инструмент,
    цикл завершается с кодом причины 'budget_steps', а не зацикливается.
    """
    from labs.lab01.starter.agent import run
    
    # Создаём задачу с очень маленьким лимитом шагов
    task = Task(
        prompt="Посчитай факториал",
        max_steps=2,  # Очень мало шагов
        max_cost_usd=10.0  # Дорого, чтобы не срабатывал по стоимости
    )
    
    # Создаём трейс в временной директории
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"AGENT_RUNS_DIR": tmpdir}):
            trace = Trace(run_id="test-max-steps")
            
            # Запускаем цикл с фиктивной моделью
            result = run(task, DEMO_TOOLS, OpenAICompatLLM(), trace)
            
            # Проверяем, что цикл завершился по правильной причине
            assert result["reason"] == "budget_steps", \
                f"Ожидалась причина 'budget_steps', получена '{result['reason']}'"
            
            # Проверяем, что количество шагов не превышает лимит
            assert result["steps"] <= task.max_steps, \
                f"Количество шагов {result['steps']} превышает max_steps {task.max_steps}"
           
            # Проверяем, что трейс был записан
            assert trace.path.exists(), "Файл трассировки не был создан"


# --------------------------------------------------------------------------- #
# Тест 2: Все четыре бюджета работают
# --------------------------------------------------------------------------- #

def test_budget_steps_limit():
    """Тест: бюджет по шагам срабатывает при превышении max_steps."""
    task = Task(prompt="Тест", max_steps=5)
    budget = Budget(task=task)
    
    # Имитируем выполнение 5 шагов
    for _ in range(5):
        budget.steps += 1
    
    # Проверяем, что exceeded() возвращает правильный код
    reason = budget.exceeded()
    assert reason == "budget_steps", f"Ожидался 'budget_steps', получен '{reason}'"


def test_budget_time_limit():
    """Тест: бюджет по времени срабатывает при превышении max_seconds."""
    task = Task(prompt="Тест", max_seconds=0.1)  # 100мс
    budget = Budget(task=task)
    
    # Ждём, пока время не превысит лимит
    import time
    time.sleep(0.15)
    
    reason = budget.exceeded()
    assert reason == "budget_time", f"Ожидался 'budget_time', получен '{reason}'"


def test_budget_tokens_limit():
    """Тест: бюджет по токенам срабатывает при превышении max_tokens."""
    task = Task(prompt="Тест", max_tokens=100)
    budget = Budget(task=task)
    
    # Имитируем использование токенов
    budget.tokens = 150
    
    reason = budget.exceeded()
    assert reason == "budget_tokens", f"Ожидался 'budget_tokens', получен '{reason}'"


def test_budget_cost_limit():
    """Тест: бюджет по стоимости срабатывает при превышении max_cost_usd."""
    task = Task(prompt="Тест", max_cost_usd=0.001)
    budget = Budget(task=task)
    
    # Имитируем расходы
    budget.cost_usd = 0.01
    
    reason = budget.exceeded()
    assert reason == "budget_cost", f"Ожидался 'budget_cost', получен '{reason}'"


# --------------------------------------------------------------------------- #
# Тест 3: Мягкий порог 80%
# --------------------------------------------------------------------------- #

def test_soft_warning_at_80_percent():
    """
    Тест: мягкий порог 80% генерирует предупреждение.
    
    Проверяется, что при достижении 80% от любого лимита
    метод soft_warning() возвращает предупреждение.
    """
    task = Task(prompt="Тест", max_steps=100)
    budget = Budget(task=task)
    
    # Достигаем 80% по шагам
    budget.steps = 80
    
    warning = budget.soft_warning()
    assert warning is not None, "Ожидалось предупреждение при 80% лимита"
    assert "80%" in warning or "бюджет" in warning.lower(), \
        f"Предупреждение не содержит ожидаемой информации: {warning}"


def test_soft_warning_only_once():
    """
    Тест: мягкое предупреждение выдаётся только один раз.
    
    Проверяется, что после первого предупреждения флаг warned=True
    и последующие вызовы не возвращают предупреждение.
    """
    task = Task(prompt="Тест", max_steps=100)
    budget = Budget(task=task)
    
    # Достигаем 80% по шагам
    budget.steps = 80
    
    # Первый вызов должен вернуть предупреждение
    warning1 = budget.soft_warning()
    assert warning1 is not None, "Первый вызов должен вернуть предупреждение"
    
    # Второй вызов не должен вернуть предупреждение
    warning2 = budget.soft_warning()
    assert warning2 is None, "Второй вызов не должен возвращать предупреждение"


def test_soft_warning_passes_to_model():
    """
    Тест: предупреждение попадает в контекст окна.
    
    Проверяется, что предупреждение из soft_warning() добавляется
    в окно диалога через build_window().
    """
    from labs.lab01.starter.agent import build_window
    
    task = Task(prompt="Тест", max_steps=100)
    budget = Budget(task=task)
    
    # Достигаем 80% по шагам
    budget.steps = 80
    
    # Создаём окно с предупреждением
    history = [{"role": "user", "content": "Начало"}]
    warning = budget.soft_warning()
    window = build_window(task, history, warning)
    
    # Проверяем, что предупреждение добавлено в окно
    warning_messages = [m for m in window if m.get("role") == "system" and "бюджет" in m.get("content", "").lower()]
    assert len(warning_messages) > 0, "Предупреждение не было добавлено в окно"


# --------------------------------------------------------------------------- #
# Тест: Полная интеграция
# --------------------------------------------------------------------------- #

def test_full_agent_run_completes_successfully():
    """
    Интеграционный тест: полный прогон агента завершается успешно.
    
    Проверяет взаимодействие всех компонентов: Task, Budget, Trace, LLM.
    """
    from labs.lab01.starter.agent import run
    
    task = Task(
        prompt="Посчитай 5 факториал",
        max_steps=3,
        max_cost_usd=1.0
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {"AGENT_RUNS_DIR": tmpdir}):
            trace = Trace(run_id="test-full-run")
            result = run(task, DEMO_TOOLS, InfiniteToolCallLLM("factorial"), trace)
            
            # Проверяем, что прогон завершился (не зациклився)
            assert result["reason"] in ["budget_steps", "done", "done_unverified"], \
                f"Неожиданная причина завершения: {result['reason']}"
            
            # Проверяем трейс
            assert trace.path.exists()
            with open(trace.path, 'r', encoding='utf-8') as f:
                records = [json.loads(line) for line in f]
            
            # Должны быть записи о шагах
            step_records = [r for r in records if r.get("t") == "step"]
            assert len(step_records) > 0, "Не найдено записей о шагах"
            
            # Должна быть запись о завершении
            finish_records = [r for r in records if r.get("t") == "finish"]
            assert len(finish_records) == 1, "Не найдена запись о завершении"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])