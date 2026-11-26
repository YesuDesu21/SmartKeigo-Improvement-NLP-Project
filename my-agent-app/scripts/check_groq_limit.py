"""Check whether the Groq rate limit has reset. Groq counts max_tokens against the limit, so asking for
2000 (about one agent call) only says READY when a real sentence would fit; it actually uses a few tokens.
    python -m scripts.check_groq_limit
Prints READY, or Groq's own message saying which limit was hit and how long to wait."""
import os
import re

from dotenv import find_dotenv, load_dotenv
from groq import Groq, RateLimitError

from src.agent.config import RunConfig

load_dotenv(find_dotenv())
model = RunConfig().model
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
try:
    client.chat.completions.create(model=model, max_tokens=2000, messages=[{"role": "user", "content": "hi"}])
    print(f"READY: {model} is accepting requests. Run the eval with --resume.")
except RateLimitError as e:
    msg = str(e)
    wait = re.search(r"try again in ([0-9hms.]+)", msg)
    kind = re.search(r"on (tokens|requests) per (day|minute)", msg)
    print(f"LIMITED: {kind.group(0) if kind else 'rate limit'} for {model}.")
    print(f"Try again in {wait.group(1)}." if wait else msg)
