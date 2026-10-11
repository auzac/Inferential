from __future__ import annotations

import json

from inferential_studio.infrastructure.settings import (
    MAX_RECENT,
    AppSettings,
    load_settings,
    save_settings,
    settings_path,
)


def test_defaults_when_missing():
    assert load_settings() == AppSettings()


def test_roundtrip_uses_isolated_home(isolated_app_home):
    s = AppSettings(display_timezone="Asia/Singapore", author_name="Audrey", log_level="DEBUG")
    save_settings(s)
    assert settings_path().is_relative_to(isolated_app_home)
    assert load_settings() == s


def test_corrupt_file_falls_back_and_is_kept_aside():
    path = settings_path()
    path.write_text("{broken", encoding="utf-8")
    assert load_settings() == AppSettings()
    assert path.with_suffix(".corrupt.json").exists()


def test_invalid_values_replaced_with_defaults():
    settings_path().write_text(
        json.dumps({"display_timezone": "Nowhere/City", "log_level": "LOUD", "unknown_key": 1}),
        encoding="utf-8",
    )
    s = load_settings()
    assert s.display_timezone == "UTC"
    assert s.log_level == "INFO"


def test_recent_projects_deduplicated_and_limited():
    s = AppSettings()
    for i in range(MAX_RECENT + 3):
        s.add_recent(f"/p/{i}.imsproj")
    s.add_recent("/p/5.imsproj")
    assert len(s.recent_projects) == MAX_RECENT
    assert s.recent_projects[0].endswith("5.imsproj")
    assert len(set(s.recent_projects)) == MAX_RECENT
