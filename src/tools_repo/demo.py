from pathlib import Path
import math


from src.agent import Tool, ToolRegistry


# --------------------------------------------------------------------------- #
# Демонстрационные инструменты
# --------------------------------------------------------------------------- #

def _factorial(n: int) -> str:
    """_summary_

    Args:
        n (int): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        str: _description_
    """
    n = int(n)
    if n < 0:
        raise ValueError("факториал определён только для n >= 0")
    if n > 2000:
        raise ValueError("n слишком велико, предел 2000")
    return str(math.factorial(n))


factorial = Tool(
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
)



def _read_file(path: str) -> str:
    """_summary_

    Args:
        path (str): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        str: _description_
    """
    p = Path(path)
    if not p.exists():
        siblings = sorted(x.name for x in (p.parent if p.parent.exists()
                                           else Path(".")).iterdir())[:10]
        raise FileNotFoundError("файла нет; рядом: " + ", ".join(siblings))
    return p.read_text(encoding="utf-8", errors="replace")

read_file = Tool(
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
    )

DEMO_TOOLS = ToolRegistry(tools=[factorial, read_file])
