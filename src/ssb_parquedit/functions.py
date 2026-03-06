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