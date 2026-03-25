import os
from pathlib import Path


def get_dapla_group() -> str:
    """Retrieves the Dapla group context from environment variables.

    Returns:
        str: The value of DAPLA_GROUP_CONTEXT, or an empty string if not set.
    """
    dapla_group: str = os.getenv("DAPLA_GROUP_CONTEXT", "")

    return dapla_group


def get_team_name() -> str:
    """Extracts the team name from the Dapla group context environment variable.

    Reads the DAPLA_GROUP_CONTEXT environment variable and extracts the team name
    by removing everything after the last '-'.
    For example, 'dapla-ffunk-developers' becomes 'dapla-ffunk'.

    Returns:
        str: The extracted team name.
    """
    s = get_dapla_group()
    team_name: str = s[: s.rfind("-")]

    return team_name


def get_bucket_name() -> str:
    """Builds a GCS bucket name based on environment variables.

    Constructs the bucket name using the team name from DAPLA_GROUP_CONTEXT
    and the DAPLA_ENVIRONMENT variable, following the format:
    ssb-{team_name}-data-produkt-{environment}.

    Returns:
        str: The constructed bucket name.
    """
    team_name: str = get_team_name()
    environment: str = os.getenv("DAPLA_ENVIRONMENT", "").lower()
    bucket_name: str = f"ssb-{team_name}-data-produkt-{environment}"

    return bucket_name


def get_dapla_environment() -> str:
    """Retrieves the Dapla environment from environment variables.

    Returns:
        str: The value of DAPLA_ENVIRONMENT in lowercase, or an empty string if not set.
    """
    environment: str = os.getenv("DAPLA_ENVIRONMENT", "").lower()
    return environment


def create_config() -> dict[str, str]:
    """Create a default database configuration dictionary.

    Generates configuration values dynamically by querying the current
    Dapla group, bucket name, and team name from the environment.

    Returns:
        dict[str, str]: A dictionary containing the following keys:
            - dbname: The name of the database.
            - dbuser: The database user, formatted as a service account email.
            - data_path: The GCS path used for temporary parquedit data.
            - catalog_name: The catalog name derived from the team name.
            - metadata_schema: The metadata schema derived from the team name.
    """
    db_config: dict[str, str] = {
        "dbname": "dapla-ffunk",
        "dbuser": f"{get_dapla_group()}@dapla-group-sa-t-57.iam",
        "data_path": f"gs://{get_bucket_name()}/.parquedit_data",
        "catalog_name": get_team_name().replace("-", "_"),
        "metadata_schema": get_team_name().replace("-", "_"),
    }

    return db_config


def repo_root_dir(directory: Path | str | None = None) -> Path:
    """Find the root directory of a git repo, searching upwards from a given path.

    Args:
        directory: The path to search from, defaults to the current working directory.
            The directory can be of type string or of type pathlib.Path.

    Returns:
        Path to the git repo's root directory.

    Raises:
        RuntimeError: If no .git directory is found when searching upwards.

    Example:
    --------
    >>> from ssb_parquedit.functions import repo_root_dir
    >>> from pathlib import Path
    >>> root = repo_root_dir()
    >>> isinstance(root, Path)
    True
    >>> (root / "pyproject.toml").exists()
    True
    """
    if directory is None:
        directory = Path.cwd()

    if isinstance(directory, str):
        directory = Path(directory)

    while directory / ".git" not in directory.iterdir():
        if directory == Path("/"):
            raise RuntimeError(f"The directory {directory} is not in a git repo.")
        else:
            directory = directory.parent
    return directory
