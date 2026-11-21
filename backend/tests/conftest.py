from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from recsys.serving.testing import make_tiny_bundle

from app.main import create_app
from app.settings import Settings


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    return make_tiny_bundle(tmp_path_factory.mktemp("bundle") / "tiny", with_forecast=True)


def make_client(bundle, **overrides) -> TestClient:
    options = {
        "database_url": "sqlite:///:memory:",
        "bundle_dir": bundle,
        "log_impressions": True,
        "auth_mode": "dev",  # most tests act as named users via X-User-Id
        "jwt_secret": "test-signing-key-test-signing-key!!",
        "password_scrypt_n": 2**10,  # fast hashing in tests
        "releases_refresh_hours": 0,  # never look for new releases on the internet during tests
        **overrides,
    }
    return TestClient(create_app(Settings(**options)))


@pytest.fixture
def client(bundle):
    with make_client(bundle) as c:
        yield c


@pytest.fixture
def local_client(bundle):
    """Real accounts (auth mode "local")."""
    with make_client(bundle, auth_mode="local", admin_emails=["ops@example.com"]) as c:
        yield c
