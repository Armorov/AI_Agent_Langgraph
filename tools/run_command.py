import subprocess
from langchain_core.tools import tool

@tool
def run_command(command: str) -> str:
    """
    Выполняет произвольную команду в терминале Linux и возвращает результат.
    """
    print(f'\nCommand {command}')
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return f"Ошибка выполнения: {result.stderr.strip()}"
        return result.stdout.strip()
    except Exception as e:
        return f"Ошибка: {str(e)}"
