from typing import Any
import os

def get_dapla_group() -> str:
    """Retrieves the Dapla group context from environment variables.

    Returns:
        str: The value of DAPLA_GROUP_CONTEXT, or an empty string if not set.
    """
    dapla_group: str = os.getenv('DAPLA_GROUP_CONTEXT', "")

    return dapla_group


def get_team_name() -> str:
    """Extracts the team name from the Dapla group context environment variable.

    Reads the DAPLA_GROUP_CONTEXT environment variable and extracts the team name
    by taking everything up to and including the second '-'.
    For example, 'dapla-ffunk-developers' becomes 'dapla-ffunk'.

    Returns:
        str: The extracted team name.
    """
    team_name: str = '-'.join(get_dapla_group().split('-')[:2])

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
    environment: str = os.getenv('DAPLA_ENVIRONMENT', "").lower()
    bucket_name: str = f"ssb-{team_name}-data-produkt-{environment}"

    return bucket_name

def create_config(short_name: str | None) -> dict[str, str]:
    
    db_config: dict[str, str] = {
        "short_name": f"{short_name}",
        "dbname": "dapla-ffunk",
        "dbuser": f"{get_dapla_group()}@dapla-group-sa-t-57.iam",
        "data_path": f"gs://{get_bucket_name()}/{short_name}/.parquedit_data",
        "catalog_name": get_team_name().replace("-", "_"),
        "metadata_schema": get_team_name().replace("-", "_"),
    }

    return db_config
