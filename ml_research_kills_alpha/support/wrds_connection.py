"""
WRDS connection helper.
Features
- Loads WRDS_USERNAME and WRDS_PASSWORD from .env file
- Creates/updates pgpass file automatically (no prompts), and sets PGPASSFILE for psycopg2.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Optional, Tuple
from contextlib import contextmanager

from ml_research_kills_alpha.support import Logger

logger = Logger()

DEFAULT_HOST = "wrds-pgdata.wharton.upenn.edu"
DEFAULT_PORT = 9737
DEFAULT_DB = "wrds"


def _load_dotenv_if_available(dotenv_path: Optional[Path] = None) -> None:
    """Load .env using python-dotenv if available. Silent if not installed."""
    try:
        from dotenv import load_dotenv  # type: ignore
        if dotenv_path is None:
            load_dotenv()  # default lookup
        else:
            load_dotenv(dotenv_path)
    except Exception:
        # Not installed or failed — it's fine if env vars are set another way.
        pass


def get_wrds_credentials(
    user: Optional[str] = None, password: Optional[str] = None
) -> Tuple[str, str]:
    """Return WRDS credentials from args or environment.

    Prefers explicit args; falls back to env vars WRDS_USERNAME/WRDS_PASSWORD.
    Raises RuntimeError if missing.
    """
    _load_dotenv_if_available()
    u = user or os.getenv("WRDS_USERNAME")
    p = password or os.getenv("WRDS_PASSWORD")
    if not u or not p:
        raise RuntimeError(
            "WRDS credentials not found. Set WRDS_USERNAME and WRDS_PASSWORD in your environment or .env."
        )
    return u, p


def _pgpass_default_path() -> Path:
    r"""Return the platform-appropriate pgpass path.

    Windows: %APPDATA%\postgresql\pgpass.conf
    POSIX:   ~/.pgpass
    """
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(appdata) / "postgresql" / "pgpass.conf"
    else:
        return Path.home() / ".pgpass"


def _format_pgpass_line(host: str, port: int, dbname: str, user: str, password: str) -> str:
    return f"{host}:{port}:{dbname}:{user}:{password}\n"


def ensure_pgpass(
    user: str,
    password: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    dbname: Optional[str] = None,
    file_path: Optional[Path] = None,
) -> Path:
    """
    Create or update a pgpass file with a single WRDS entry.

    - Ensures directory exists.
    - Replaces any existing line that matches host:port:dbname:user.
    - Applies 0600 permissions on POSIX (required by libpq).
    - Sets PGPASSFILE to the resulting path for this process.
    """
    host = host or os.getenv("WRDS_PGHOST", DEFAULT_HOST)
    port = int(port or os.getenv("WRDS_PGPORT", DEFAULT_PORT))
    dbname = dbname or os.getenv("WRDS_DBNAME", DEFAULT_DB)

    pg_file = file_path or _pgpass_default_path()
    pg_file.parent.mkdir(parents=True, exist_ok=True)

    key_prefix = f"{host}:{port}:{dbname}:{user}:"
    new_line = _format_pgpass_line(host, port, dbname, user, password)

    if pg_file.exists():
        try:
            lines = pg_file.read_text().splitlines(True)  # keepends
        except UnicodeDecodeError:
            lines = []
        # Remove existing matching entry
        lines = [ln for ln in lines if not ln.startswith(key_prefix)]
    else:
        lines = []

    lines.append(new_line)
    pg_file.write_text("".join(lines))

    # Required permissions on POSIX systems
    if os.name != "nt":
        try:
            pg_file.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except Exception as e:
            logger.debug("Failed to chmod pgpass file: %s", e)

    os.environ["PGPASSFILE"] = str(pg_file)
    logger.debug(f"PGPASSFILE set to {pg_file}")
    return pg_file


def connect_wrds(
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    dbname: Optional[str] = None,
    create_pgpass: bool = True,
    **wrds_kwargs,
):
    """
    Return an authenticated wrds.Connection without interactive prompts.

    Args:
        user/password: Optional overrides; else read from env/.env.
        host/port/dbname: Optional overrides; else env or defaults.
        create_pgpass: If True, create/update pgpass and set PGPASSFILE.
        **wrds_kwargs: Passed through to wrds.Connection(...).

    The function avoids any CLI prompts by ensuring credentials and pgpass are set up first.
    """
    _load_dotenv_if_available()

    # Resolve settings
    host = host or os.getenv("WRDS_PGHOST", DEFAULT_HOST)
    port = int(port or os.getenv("WRDS_PGPORT", DEFAULT_PORT))
    dbname = dbname or os.getenv("WRDS_DBNAME", DEFAULT_DB)

    u, p = get_wrds_credentials(user, password)

    if create_pgpass:
        ensure_pgpass(u, p, host=host, port=port, dbname=dbname)

    # Finally, connect; rely on pgpass or pass creds explicitly.
    import wrds  # type: ignore
    os.environ["WRDS_USERNAME"] = u
    os.environ["WRDS_PASSWORD"] = p
    os.environ.setdefault("PGUSER", u)
    if not create_pgpass:
        # Only set PGPASSWORD if we are not using pgpass; otherwise libpq prefers pgpass.
        os.environ["PGPASSWORD"] = p
    conn = wrds.Connection(wrds_username=u, wrds_password=p, **wrds_kwargs)

    logger.info(f"Connected to WRDS at {host}:{port}/{dbname} as {u}")
    return conn


@contextmanager
def wrds_connection(**kwargs):
    """Context manager wrapper around connect_wrds().

    Example:
        with wrds_connection() as conn:
            df = conn.get_table('crsp', 'msf', obs=1)
    """
    conn = connect_wrds(**kwargs)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def prepare_wrds_noninteractive(
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    dbname: Optional[str] = None,
) -> Path:
    """
    Ensure WRDS will *never* prompt in this process.

    - Loads creds from args or env/.env.
    - Creates/updates pgpass and exports PGPASSFILE.
    - Exports WRDS_USERNAME/WRDS_PASSWORD/PGUSER/PGPASSWORD env vars.

    Returns the pgpass path used.
    """
    _load_dotenv_if_available()
    host = host or os.getenv("WRDS_PGHOST", DEFAULT_HOST)
    port = int(port or os.getenv("WRDS_PGPORT", DEFAULT_PORT))
    dbname = dbname or os.getenv("WRDS_DBNAME", DEFAULT_DB)
    u, p = get_wrds_credentials(user, password)
    pgpass_path = ensure_pgpass(u, p, host=host, port=port, dbname=dbname)
    os.environ["WRDS_USERNAME"] = u
    os.environ["WRDS_PASSWORD"] = p
    os.environ["PGUSER"] = u
    os.environ["PGPASSWORD"] = p
    os.environ["PGPASSFILE"] = str(pgpass_path)
    logger.debug(f"WRDS non-interactive env prepared: {pgpass_path}")
    return pgpass_path


__all__ = [
    "connect_wrds",
    "wrds_connection",
    "ensure_pgpass",
    "get_wrds_credentials",
    "prepare_wrds_noninteractive",
]


if __name__ == "__main__":
    # Minimal smoke test when run directly.
    # Requires WRDS_USERNAME/WRDS_PASSWORD in env or .env.
    import pandas as pd  # noqa: F401

    with wrds_connection():
        print("WRDS connection successful.")
