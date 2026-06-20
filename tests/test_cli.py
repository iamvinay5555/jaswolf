from jaswolf.cli import _redact, main


def test_redact_masks_url_credentials():
    assert _redact("postgresql://jaswolf:s3cretpw@host:5432/db") == "postgresql://jaswolf:***@host:5432/db"
    assert _redact("redis://:onlypass@host:6379/0") == "redis://:***@host:6379/0"
    assert _redact("sqlite:///./jaswolf.db") == "sqlite:///./jaswolf.db"
    assert _redact("http://host:8000/v1") == "http://host:8000/v1"  # no creds untouched
    assert _redact(None) == "(unset)"
    assert _redact("") == "(unset)"


def test_diagnose_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JASWOLF_DATABASE_URL", f"sqlite:///{tmp_path}/diag.db")
    monkeypatch.setenv("JASWOLF_EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("JASWOLF_LOG_LEVEL", "WARNING")
    main(["diagnose", "--user-id", "alice"])
    out = capsys.readouterr().out
    assert "## JasWolf diagnostic report" in out
    assert "storage: sqlite" in out
    assert "live probe" in out
    assert "thresholds:" in out
