from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()
def pick_llm(level:str):
    if level.lower() == "low":
        llm = ChatOpenRouter(model="stealth/space-bunny-alpha",temperature=0)
    elif level.lower() == "medium":
        llm = ChatOpenRouter(model="stealth/space-bunny-alpha",temperature=0)
    elif level.lower() == "high":
        llm = ChatOpenRouter(model="stealth/space-bunny-alpha",temperature=0)
    else:
        raise ValueError(f"Unsupported level : {level}")
    return llm

    if __name__ == "__main__":
        llm = pick_llm("low")
        print(llm.invoke("Hello, how are you?"))