"""Cloud SQL "manual IAM database authentication" -- direct connection.

This is an alternative to both the Cloud SQL Auth Proxy (subprocess) and the
internals-based ``cloud-sql-python-connector`` relay: no local proxy, no
relay thread, no reach-in to private Connector attributes. DuckDB connects
straight to the instance's IP address over TLS, using a short-lived OAuth
2.0 access token as the Postgres password. This is the same mechanism
``gcloud sql generate-login-token`` implements, and it is officially
documented by Google as usable for direct connections (no Proxy/Connector
required): https://docs.cloud.google.com/sql/docs/postgres/iam-logins

One-time setup required on the Cloud SQL side (done by an admin, not by
this code):

- The instance must have the ``cloudsql.iam_authentication`` database flag
  set to ``on``.
- The IAM principal used to connect (user or service account) must exist as
  a Cloud SQL database user of type ``CLOUD_IAM_USER`` /
  ``CLOUD_IAM_SERVICE_ACCOUNT``, and hold the ``roles/cloudsql.instanceUser``
  and ``roles/cloudsql.client`` IAM roles.
- For a service account, the *database username* is its email address with
  the trailing ``.gserviceaccount.com`` stripped -- this is a Cloud SQL
  requirement. Make sure ``dbuser`` is passed in already in that form.

Networking: this still requires the runtime to be able to reach the
instance's IP directly (same requirement the Cloud SQL Proxy sidecar has
today for a private-IP instance -- VPC peering / Serverless VPC Access /
etc.). This module does not change networking, only how the connection is
authenticated.

Caveats:

- Access tokens are valid for ~1 hour. A fresh token is fetched per
  ``DuckDBConnection``/``ParquEdit`` instantiation, which is fine for
  short-lived scripts and jobs. A service that holds a single connection
  open for many hours risks the token expiring if DuckDB internally
  reopens the Postgres connection later -- for that use case, recreate the
  connection periodically rather than relying on this module to refresh
  mid-session.
"""

from __future__ import annotations

import logging
from typing import cast

import google.auth
import google.auth.transport.requests
from google.auth.credentials import Credentials

logger = logging.getLogger(__name__)

_LOGIN_SCOPE = "https://www.googleapis.com/auth/sqlservice.login"
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_SQLADMIN_API = "https://sqladmin.googleapis.com/sql/v1beta4"


def get_login_token(credentials: Credentials | None = None) -> str:
    """Fetch a short-lived OAuth 2.0 access token for Cloud SQL IAM login.

    Equivalent to ``gcloud sql generate-login-token``. The returned token is
    used directly as the Postgres password and is valid for roughly one
    hour.

    Args:
        credentials: Optional pre-built credentials. If omitted, Application
            Default Credentials are fetched, scoped to ``sqlservice.login``.

    Returns:
        A short-lived OAuth 2.0 access token string.

    Raises:
        RuntimeError: If no token could be obtained.
    """
    if credentials is None:
        credentials, _ = google.auth.default(scopes=[_LOGIN_SCOPE])

    credentials.refresh(google.auth.transport.requests.Request())  # type: ignore[no-untyped-call]
    token = cast("str | None", credentials.token)
    if not token:
        raise RuntimeError("Failed to obtain a Cloud SQL IAM login token.")
    return token


def get_instance_ip(
    instance_connection_name: str,
    *,
    ip_type: str = "PRIVATE",
    credentials: Credentials | None = None,
) -> str:
    """Look up a Cloud SQL instance's IP address via the Cloud SQL Admin API.

    Uses the public, stable ``sqladmin`` REST API (``instances.get``)
    directly -- no private/internal attributes of any client library are
    touched, unlike the ``cloud-sql-python-connector``-internals relay.

    Args:
        instance_connection_name: ``project:region:instance``, e.g.
            ``my-gcp-project:europe-north1:my-db-instance``.
        ip_type: IP address type to select, matching the ``type`` field
            returned by the Cloud SQL Admin API for each configured
            address: ``"PRIVATE"``, ``"PRIMARY"`` (public IP), or
            ``"OUTGOING"``.
        credentials: Optional pre-built credentials. If omitted, Application
            Default Credentials are fetched, scoped for ``cloud-platform``.

    Returns:
        The instance's IP address matching ``ip_type``.

    Raises:
        ValueError: If ``instance_connection_name`` is malformed, or the
            instance has no IP address of the requested type.
    """
    parts = instance_connection_name.split(":")
    if len(parts) != 3:
        raise ValueError(
            "instance_connection_name must be 'project:region:instance', "
            f"got {instance_connection_name!r}"
        )
    project, _region, instance = parts

    if credentials is None:
        credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
    session = google.auth.transport.requests.AuthorizedSession(credentials)  # type: ignore[no-untyped-call]
    resp = session.get(f"{_SQLADMIN_API}/projects/{project}/instances/{instance}")
    resp.raise_for_status()
    ip_addresses = resp.json().get("ipAddresses", [])

    for entry in ip_addresses:
        if entry.get("type") == ip_type:
            return str(entry["ipAddress"])

    available = [entry.get("type") for entry in ip_addresses]
    raise ValueError(
        f"Instance {instance_connection_name!r} has no IP address of type "
        f"{ip_type!r}. Available types: {available}"
    )