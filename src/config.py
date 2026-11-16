import os
from dotenv import load_dotenv

# Load environment variables from a local .env file when one is present.
# Missing keys are not fatal at import time so tests and tooling can import
# this module with dummy values or with no credentials at all.
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")


def require_api_keys():
    """Return Alpaca credentials, or raise if they were never configured.

    Call this at the point a live client is constructed. Importing ``config``
    must stay safe when keys are absent.
    """
    missing = []
    if not API_KEY:
        missing.append("ALPACA_API_KEY")
    if not API_SECRET:
        missing.append("ALPACA_SECRET_KEY")
    if missing:
        raise ValueError(
            "Missing Alpaca API keys: "
            + ", ".join(missing)
            + ". Set them in the environment or a .env file. "
            "Tests should set dummy values and mock the network client."
        )
    return API_KEY, API_SECRET
