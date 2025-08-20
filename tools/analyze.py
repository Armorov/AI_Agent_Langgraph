from langchain_core.tools import tool

@tool
def analyze(path = None):
    '''Производит анализ кода в указанной папке'''
    if not path:
        pass
    return f"заглушка"