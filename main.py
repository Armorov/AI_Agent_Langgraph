import asyncio
from assistant import Assistant

async def main():
    assistant = Assistant()

    # Приветственное сообщение и список инструментов
    print("Ассистент: Привет! Я твой помощник. Список доступных команд:")
    for t in assistant.tools:
        print(f" - {t.name}: {t.description}")

    while True:
        user_input = input("Вы: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["выход", "quit", "exit"]:
            print("Ассистент: До свидания!")
            break

        # Запуск анимации
        task = asyncio.create_task(assistant.handle_input(user_input))
        await task

if __name__ == "__main__":
    asyncio.run(main())
