"""
Doctor script for checking project environment and dependencies.

Run: python tools/doctor.py

Checks:
1. Project state: Python version, Docker, environment variables, write permissions
2. Dependencies: Required packages, model availability
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any


def _load_env_file(env_path: str = ".env") -> None:
    """Load environment variables from .env file without external dependencies.
    
    Simple parser that handles:
    - KEY=value
    - KEY="value"
    - KEY='value'
    - Comments starting with #
    """
    env_file = Path(env_path)
    if not env_file.exists():
        return
    
    content = env_file.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Parse KEY=value
        match = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$', line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()
            
            # Remove quotes
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            # Only set if not already set
            if key not in os.environ:
                os.environ[key] = value


# Load .env file if it exists
_load_env_file()


# =============================================================================
# SOLID-compliant refactoring of check_project_state()
# =============================================================================

class CheckResult:
    """Represents the result of a single check."""
    
    def __init__(self, name: str, status: str, message: str):
        self.name = name
        self.status = status
        self.message = message
    
    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message
        }


class Check(ABC):
    """Abstract base class for all checks (Interface Segregation Principle)."""
    
    @abstractmethod
    def run(self) -> CheckResult:
        """Execute the check and return the result."""
        pass


class PythonVersionCheck(Check):
    """Check Python version requirement (3.11+)."""
    
    def run(self) -> CheckResult:
        py_version = sys.version_info
        py_ok = py_version >= (3, 11)
        
        if py_ok:
            return CheckResult(
                name="Python version",
                status="OK",
                message=f"Python {py_version.major}.{py_version.minor}.{py_version.micro}"
            )
        else:
            return CheckResult(
                name="Python version",
                status="FAIL",
                message=f"Python {py_version.major}.{py_version.minor}.{py_version.micro}"
            )


class DockerCheck(Check):
    """Check Docker availability and functionality."""
    
    def run(self) -> CheckResult:
        docker_path = shutil.which("docker")
        
        if not docker_path:
            return CheckResult(
                name="Docker",
                status="SKIP",
                message="Docker не установлен (необязательно для базовой работы)"
            )
        
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            docker_ok = result.returncode == 0
            
            if docker_ok:
                return CheckResult(
                    name="Docker",
                    status="OK",
                    message="Docker доступен и работает"
                )
            else:
                return CheckResult(
                    name="Docker",
                    status="FAIL",
                    message="Docker установлен, но не работает"
                )
        except (subprocess.TimeoutExpired, Exception) as e:
            return CheckResult(
                name="Docker",
                status="FAIL",
                message=f"Ошибка проверки Docker: {e}"
            )


class ApiKeyCheck(Check):
    """Check OPENAI_API_KEY environment variable."""
    
    def run(self) -> CheckResult:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        key_ok = bool(api_key and api_key.strip())
        
        if key_ok:
            return CheckResult(
                name="OPENAI_API_KEY",
                status="OK",
                message="Ключ API установлен"
            )
        else:
            return CheckResult(
                name="OPENAI_API_KEY",
                status="FAIL",
                message="OPENAI_API_KEY не установлен"
            )


class WritePermissionsCheck(Check):
    """Check write permissions to runs/ directory."""
    
    def run(self) -> CheckResult:
        runs_dir = Path(os.environ.get("AGENT_RUNS_DIR", "runs"))
        
        try:
            runs_dir.mkdir(parents=True, exist_ok=True)
            test_file = runs_dir / ".doctor_test"
            test_file.write_text("test")
            test_file.unlink()
            
            return CheckResult(
                name="Права на запись в runs/",
                status="OK",
                message=f"Можно писать в {runs_dir}"
            )
        except Exception as e:
            return CheckResult(
                name="Права на запись в runs/",
                status="FAIL",
                message=f"Ошибка записи в {runs_dir}: {e}"
            )


class ProjectStateChecker:
    """
    Orchestrates project state checks.
    
    Follows Single Responsibility Principle - only responsible for
    coordinating checks and aggregating results.
    Follows Dependency Inversion Principle - depends on abstract Check class.
    """
    
    def __init__(self, checks: list[Check] | None = None):
        """
        Initialize with optional list of checks.
        
        Args:
            checks: List of Check instances. If None, uses default checks.
        """
        self._checks = checks or self._default_checks()
    
    def _default_checks(self) -> list[Check]:
        """Return the default list of checks."""
        return [
            PythonVersionCheck(),
            DockerCheck(),
            ApiKeyCheck(),
            WritePermissionsCheck(),
        ]
    
    def run_all(self) -> dict[str, Any]:
        """
        Run all checks and return aggregated results.
        
        Returns:
            dict with 'ok' (bool), 'checks' (list of check results), 'errors' (list of error messages)
        """
        checks = []
        errors = []
        all_ok = True
        
        for check in self._checks:
            result = check.run()
            checks.append(result.to_dict())
            
            if result.status == "FAIL":
                all_ok = False
                errors.append(self._format_error(result))
        
        return {
            "ok": all_ok,
            "checks": checks,
            "errors": errors
        }
    
    def _format_error(self, result: CheckResult) -> str:
        """Format error message based on check result."""
        if result.name == "Python version":
            py_version = sys.version_info
            return f"Python 3.11+ требуется, найден {py_version.major}.{py_version.minor}"
        elif result.name == "Docker":
            return "Docker установлен, но не работает"
        elif result.name == "OPENAI_API_KEY":
            return "OPENAI_API_KEY не установлен в переменных окружения"
        elif result.name == "Права на запись в runs/":
            return result.message
        return result.message


def check_project_state() -> dict[str, Any]:
    """Check project state: Python version, Docker, environment variables, write permissions.
    
    Returns:
        dict with 'ok' (bool), 'checks' (list of check results), 'errors' (list of error messages)
    """
    checker = ProjectStateChecker()
    return checker.run_all()


# =============================================================================
# SOLID-compliant refactoring of check_dependencies()
# =============================================================================

# Mapping from package names to module names for import
PACKAGE_TO_MODULE = {
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
    "sqlite-vec": "sqlite_vec",
}


def _parse_requirements(requirements_path: str = "requirements.txt") -> list[tuple[str, str]]:
    """
    Parse requirements.txt and return list of (module_name, package_spec) tuples.
    
    Args:
        requirements_path: Path to requirements.txt file
        
    Returns:
        List of tuples (module_name, package_spec)
    """
    req_file = Path(requirements_path)
    if not req_file.exists():
        return []
    
    packages = []
    content = req_file.read_text(encoding="utf-8")
    
    for line in content.splitlines():
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        
        # Remove inline comments
        if "#" in line:
            line = line.split("#")[0].strip()
        
        if not line:
            continue
        
        # Extract package spec (e.g., "openai>=1.40")
        package_spec = line
        
        # Extract package name (before version specifier)
        match = re.match(r'^([a-zA-Z0-9_-]+)', package_spec)
        if match:
            package_name = match.group(1)
            # Map to module name
            module_name = PACKAGE_TO_MODULE.get(package_name, package_name)
            packages.append((module_name, package_spec))
    
    return packages


def _parse_version_spec(spec: str) -> tuple[str, str | None, str | None]:
    """
    Parse a version specification into (package_name, operator, version).
    
    Examples:
        "openai>=1.40" -> ("openai", ">=", "1.40")
        "pydantic>=2.7" -> ("pydantic", ">=", "2.7")
        "numpy>=1.26" -> ("numpy", ">=", "1.26")
        "pytest>=8.2" -> ("pytest", ">=", "8.2")
        "package" -> ("package", None, None)
    """
    match = re.match(r'^([a-zA-Z0-9_-]+)(>=|<=|>|<|==|!=)?(.+)?$', spec.strip())
    if match:
        name = match.group(1)
        op = match.group(2)
        ver = match.group(3)
        return (name, op, ver)
    return (spec, None, None)


def _compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.
    
    Returns:
        -1 if v1 < v2
         0 if v1 == v2
         1 if v1 > v2
    """
    def normalize(v):
        return [int(x) for x in re.sub(r'[^\d.]', '', v).split('.')]
    
    parts1 = normalize(v1)
    parts2 = normalize(v2)
    
    # Pad with zeros to make equal length
    max_len = max(len(parts1), len(parts2))
    parts1.extend([0] * (max_len - len(parts1)))
    parts2.extend([0] * (max_len - len(parts2)))
    
    for p1, p2 in zip(parts1, parts2):
        if p1 < p2:
            return -1
        elif p1 > p2:
            return 1
    return 0


def _version_satisfies(installed: str, operator: str | None, required: str | None) -> bool:
    """
    Check if installed version satisfies the requirement.
    
    Args:
        installed: The installed version string
        operator: The version operator (>=, <=, >, <, ==, !=) or None
        required: The required version string or None
    
    Returns:
        True if the installed version satisfies the requirement
    """
    if operator is None or required is None:
        return True
    
    cmp = _compare_versions(installed, required)
    
    if operator == '>=':
        return cmp >= 0
    elif operator == '<=':
        return cmp <= 0
    elif operator == '>':
        return cmp > 0
    elif operator == '<':
        return cmp < 0
    elif operator == '==':
        return cmp == 0
    elif operator == '!=':
        return cmp != 0
    
    return True


class PackageCheck(Check):
    """Check if a Python package is installed and meets version requirements."""
    
    def __init__(self, module_name: str, package_spec: str):
        self._module_name = module_name
        self._package_spec = package_spec
    
    def run(self) -> CheckResult:
        # Parse the package spec to get package name and version requirement
        pkg_name, operator, required_version = _parse_version_spec(self._package_spec)
        
        # First, check if the module can be imported
        try:
            __import__(self._module_name)
        except ImportError:
            return CheckResult(
                name=f"Пакет {self._package_spec}",
                status="FAIL",
                message="Не установлен"
            )
        
        # Try to get the installed version
        try:
            installed_version = version(pkg_name)
        except PackageNotFoundError:
            # Package not found via metadata, but module imports - OK
            return CheckResult(
                name=f"Пакет {self._package_spec}",
                status="OK",
                message="Установлен"
            )
        
        # Check version compatibility
        if operator and required_version:
            if _version_satisfies(installed_version, operator, required_version):
                return CheckResult(
                    name=f"Пакет {self._package_spec}",
                    status="OK",
                    message=f"Установлен (версия {installed_version}, требуется {operator}{required_version})"
                )
            else:
                return CheckResult(
                    name=f"Пакет {self._package_spec}",
                    status="FAIL",
                    message=f"Версия {installed_version} не удовлетворяет требованию {operator}{required_version}"
                )
        
        # No version requirement, just check if installed
        return CheckResult(
            name=f"Пакет {self._package_spec}",
            status="OK",
            message=f"Установлен (версия {installed_version})"
        )


class RequiredPackagesCheck(Check):
    """Check all required packages."""
    
    def __init__(self, packages: list[tuple[str, str]] | None = None):
        self._packages = packages if packages is not None else self._default_packages()
    
    def _default_packages(self) -> list[tuple[str, str]]:
        """Return the default list of required packages from requirements.txt."""
        return _parse_requirements("requirements.txt")
    
    def run(self) -> CheckResult:
        """Run all package checks and aggregate results."""
        checks = []
        errors = []
        all_ok = True
        
        for module_name, package_spec in self._packages:
            check = PackageCheck(module_name, package_spec)
            result = check.run()
            checks.append(result.to_dict())
            
            if result.status == "FAIL":
                all_ok = False
                # Include package name in error message
                errors.append(f"{result.name}: {result.message}")
        
        # Store results for later retrieval
        self._last_checks = checks
        self._last_errors = errors
        self._last_ok = all_ok
        
        # Return a summary result
        if all_ok:
            return CheckResult(
                name="Зависимости",
                status="OK",
                message="Все пакеты установлены"
            )
        else:
            return CheckResult(
                name="Зависимости",
                status="FAIL",
                message=f"Обнаружено {len(errors)} проблем с пакетами"
            )
    
    def get_checks(self) -> list[dict[str, str]]:
        """Get individual check results."""
        return getattr(self, '_last_checks', [])
    
    def get_errors(self) -> list[str]:
        """Get error messages."""
        return getattr(self, '_last_errors', [])
    
    def is_ok(self) -> bool:
        """Check if all packages are installed."""
        return getattr(self, '_last_ok', True)


class ModelCheck(Check):
    """Check model availability via OpenAI API."""
    
    def __init__(self, model: str | None = None):
        self._model = model or os.environ.get("AGENT_MODEL_MAIN", "")
    
    def run(self) -> CheckResult:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        
        if not api_key or not api_key.strip():
            return CheckResult(
                name="Модель",
                status="SKIP",
                message="Пропущена (нет API ключа)"
            )
        
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=base_url if base_url else None,
                api_key=api_key
            )
            # Minimal request to check model availability
            resp = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "рассчитай два и два"}],
                max_tokens=10,
                timeout=10
            )
            # Verify response is valid
            if resp is None or not resp.choices:
                raise Exception("Empty response from model")
            # Get the answer to verify model is working
            answer = resp.choices[0].message.content
            return CheckResult(
                name="Модель",
                status="OK",
                message=f"Модель {self._model} доступна (ответ: {answer})"
            )
        except Exception as e:
            return CheckResult(
                name="Модель",
                status="FAIL",
                message=f"Ошибка доступа к модели: {e}"
            )


class DependenciesChecker:
    """
    Orchestrates dependency checks.
    
    Follows Single Responsibility Principle - only responsible for
    coordinating checks and aggregating results.
    Follows Dependency Inversion Principle - depends on abstract Check class.
    """
    
    def __init__(self, checks: list[Check] | None = None):
        """
        Initialize with optional list of checks.
        
        Args:
            checks: List of Check instances. If None, uses default checks.
        """
        self._checks = checks or self._default_checks()
    
    def _default_checks(self) -> list[Check]:
        """Return the default list of checks."""
        return [
            RequiredPackagesCheck(),
            ModelCheck(),
        ]
    
    def run_all(self) -> dict[str, Any]:
        """
        Run all checks and return aggregated results.
        
        Returns:
            dict with 'ok' (bool), 'checks' (list of check results), 'errors' (list of error messages)
        """
        checks = []
        errors = []
        all_ok = True
        
        for check in self._checks:
            result = check.run()
            
            # Handle special case for RequiredPackagesCheck which has multiple results
            if isinstance(check, RequiredPackagesCheck):
                checks.extend(check.get_checks())
                errors.extend(check.get_errors())
                if not check.is_ok():
                    all_ok = False
            else:
                checks.append(result.to_dict())
                if result.status == "FAIL":
                    all_ok = False
                    errors.append(f"Модель {check._model} недоступна: {result.message}")
                elif result.status == "SKIP":
                    # Model skip doesn't affect overall status
                    pass
        
        return {
            "ok": all_ok,
            "checks": checks,
            "errors": errors
        }


def check_dependencies() -> dict[str, Any]:
    """Check project dependencies: required packages, model availability.
    
    Returns:
        dict with 'ok' (bool), 'checks' (list of check results), 'errors' (list of error messages)
    """
    checker = DependenciesChecker()
    return checker.run_all()


def _describe_runtime() -> str:
    """Return a concise description of the active Python environment."""
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        return venv_path

    executable = Path(sys.executable).resolve()
    project_root = Path(__file__).resolve().parent.parent
    project_venv = project_root / ".venv" / "bin" / ("python.exe" if sys.platform == "win32" else "python")
    if executable == project_venv or str(executable).startswith(str(project_root / ".venv")):
        return str(project_venv)

    return "не активирован"


def main() -> int:
    """Run all checks and print results."""
    runtime = _describe_runtime()
    print("=" * 60)
    print("Проверка окружения проекта (make doctor)")
    print(f"Python: {sys.executable}")
    print(f"Venv: {runtime}")
    print("=" * 60)
    
    # Check project state
    print("\n[1] Состояние проекта:")
    print("-" * 40)
    state_result = check_project_state()
    for check in state_result["checks"]:
        status_symbol = "✓" if check["status"] == "OK" else ("○" if check["status"] == "SKIP" else "✗")
        print(f"  {status_symbol} {check['name']}: {check['message']}")
    
    # Check dependencies
    print("\n[2] Зависимости:")
    print("-" * 40)
    deps_result = check_dependencies()
    for check in deps_result["checks"]:
        status_symbol = "✓" if check["status"] == "OK" else ("○" if check["status"] == "SKIP" else "✗")
        print(f"  {status_symbol} {check['name']}: {check['message']}")
    
    # Summary
    print("\n" + "=" * 60)
    all_ok = state_result["ok"] and deps_result["ok"]
    if all_ok:
        print(f"Результат: ✓ Всё ОК, можно продолжать (venv: {runtime})")
        print("=" * 60)
        return 0
    else:
        print(f"Результат: ✗ Обнаружены проблемы (venv: {runtime}):")
        for error in state_result["errors"] + deps_result["errors"]:
            print(f"  - {error}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())