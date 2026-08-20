from dotenv import load_dotenv, dotenv_values
from openai import OpenAI
from pathlib import Path

def make_openai_client() -> OpenAI:
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"

    if not env_path.exists():
        raise RuntimeError(f".env nu există la: {env_path}")

    load_dotenv(env_path, override=True)
    env = dotenv_values(env_path)

    # DEBUG: vezi ce chei există în fișier
    print("Folosesc .env din:", env_path)
    print("Chei găsite în .env:", list(env.keys()))

    api_key = (env.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY lipsește sau e gol în .env. Trebuie: OPENAI_API_KEY=sk-proj-...")

    print("Prefix cheie:", api_key[:12])
    return OpenAI(api_key=api_key)



