import importlib
import pkgutil
import tools

def load_tools():
    tool_list = []
    for _, module_name, _ in pkgutil.iter_modules(tools.__path__):
        module = importlib.import_module(f"tools.{module_name}")
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if hasattr(obj, "name") and hasattr(obj, "description"):
                tool_list.append(obj)
    return tool_list
