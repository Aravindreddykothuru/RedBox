"""Real accounts (auth mode "local"): sign-up, sign-in, tokens, revocation, rate limiting, data rights."""

from __future__ import annotations

from tests.conftest import make_client

PASSWORD = "correct horse battery"


def register(client, email="asha@example.com", password=PASSWORD, name="Asha"):
    r = client.post("/auth/register", json={"email": email, "password": password, "display_name": name})
    assert r.status_code == 201, r.text
    return r.json()


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_register_login_and_me(local_client):
    body = register(local_client, email="  Asha@Example.com ")
    assert body["user"]["email"] == "asha@example.com" and body["user"]["display_name"] == "Asha"
    me = local_client.get("/auth/me", headers=auth(body["access_token"])).json()
    assert me["email"] == "asha@example.com"
    login = local_client.post("/auth/login", json={"email": "ASHA@example.com", "password": PASSWORD})
    assert login.status_code == 200 and login.json()["user"]["user_id"] == body["user"]["user_id"]


def test_registration_validation_and_duplicates(local_client):
    register(local_client)
    dup = local_client.post("/auth/register", json={"email": "ASHA@example.com", "password": PASSWORD})
    assert dup.status_code == 409
    assert (
        local_client.post("/auth/register", json={"email": "x@y.z", "password": "short"}).status_code == 422
    )
    assert (
        local_client.post("/auth/register", json={"email": "not-an-email", "password": PASSWORD}).status_code
        == 422
    )


def test_wrong_password_and_unknown_email_look_the_same(local_client):
    register(local_client)
    wrong = local_client.post("/auth/login", json={"email": "asha@example.com", "password": "nope-nope"})
    unknown = local_client.post("/auth/login", json={"email": "nobody@example.com", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()  # no hint whether the account exists


def test_personal_endpoints_need_a_valid_token(local_client):
    token = register(local_client)["access_token"]
    assert local_client.get("/me/ratings").status_code == 401
    assert (
        local_client.get("/me/ratings", headers={"X-User-Id": "alice"}).status_code == 401
    )  # dev header ignored
    assert local_client.get("/me/ratings", headers=auth(token[:-2] + "xx")).status_code == 401  # tampered
    assert local_client.put("/ratings/100", json={"rating": 5}, headers=auth(token)).status_code == 200
    assert [r["movie_id"] for r in local_client.get("/me/ratings", headers=auth(token)).json()] == [100]
    assert local_client.get("/movies/100").status_code == 200  # browsing needs no account


def test_users_only_see_their_own_data(local_client):
    a = register(local_client, email="a@example.com")["access_token"]
    b = register(local_client, email="b@example.com")["access_token"]
    local_client.put("/ratings/101", json={"rating": 4}, headers=auth(a))
    assert local_client.get("/me/ratings", headers=auth(b)).json() == []


def test_password_change_revokes_old_tokens(local_client):
    old = register(local_client)["access_token"]
    bad = local_client.post(
        "/auth/password",
        json={"current_password": "wrong-one", "new_password": "a new password"},
        headers=auth(old),
    )
    assert bad.status_code == 403
    r = local_client.post(
        "/auth/password",
        json={"current_password": PASSWORD, "new_password": "a new password"},
        headers=auth(old),
    )
    assert r.status_code == 200
    assert local_client.get("/auth/me", headers=auth(old)).status_code == 401  # signed out everywhere
    assert local_client.get("/auth/me", headers=auth(r.json()["access_token"])).status_code == 200
    assert (
        local_client.post("/auth/login", json={"email": "asha@example.com", "password": PASSWORD}).status_code
        == 401
    )
    assert (
        local_client.post(
            "/auth/login", json={"email": "asha@example.com", "password": "a new password"}
        ).status_code
        == 200
    )


def test_logout_everywhere(local_client):
    token = register(local_client)["access_token"]
    assert local_client.post("/auth/logout-all", headers=auth(token)).status_code == 204
    assert local_client.get("/auth/me", headers=auth(token)).status_code == 401


def test_failed_logins_are_rate_limited(bundle):
    with make_client(bundle, auth_mode="local", login_max_failures=3) as c:
        register(c)
        for _ in range(3):
            assert (
                c.post("/auth/login", json={"email": "asha@example.com", "password": "wrong!!!"}).status_code
                == 401
            )
        blocked = c.post("/auth/login", json={"email": "asha@example.com", "password": PASSWORD})
        assert blocked.status_code == 429  # even the right password waits out the window


def test_profile_update_export_and_delete(local_client):
    token = register(local_client)["access_token"]
    h = auth(token)
    assert local_client.patch("/me/profile", json={"display_name": "Asha R"}, headers=h).status_code == 204
    assert local_client.get("/auth/me", headers=h).json()["display_name"] == "Asha R"
    local_client.put("/ratings/102", json={"rating": 4.5}, headers=h)
    local_client.put("/watchlist/103", headers=h)
    data = local_client.get("/me/export", headers=h).json()
    assert data["account"]["email"] == "asha@example.com"
    assert [r["movie_id"] for r in data["ratings"]] == [102] and data["watchlist"][0]["movie_id"] == 103
    assert any(e["type"] == "rate" for e in data["events"])

    assert local_client.post("/me/delete", json={"password": "wrong"}, headers=h).status_code == 403
    assert local_client.post("/me/delete", json={"password": PASSWORD}, headers=h).status_code == 204
    assert local_client.get("/auth/me", headers=h).status_code == 401
    assert (
        local_client.post("/auth/login", json={"email": "asha@example.com", "password": PASSWORD}).status_code
        == 401
    )
    again = register(local_client)["access_token"]  # the email is free again, with no old data
    assert local_client.get("/me/ratings", headers=auth(again)).json() == []


def test_account_endpoints_are_off_in_other_modes(client):
    r = client.post("/auth/register", json={"email": "a@example.com", "password": PASSWORD})
    assert r.status_code == 404
