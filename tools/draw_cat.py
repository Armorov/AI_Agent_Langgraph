from langchain_core.tools import tool

@tool
def draw_cat() -> str:
    """Рисует ASCII-котика в терминале Linux"""
    cat = r"""
         /\_/\  
        ( o.o ) 
         > ^ <  
    """
    print(cat)
    return "Котик нарисован!"