from django.conf import settings


def test_corpus_dir_exists():
    assert (settings.CORPUS_DIR / "manifest.json").is_file()


def test_handbook_app_installed():
    assert "webapp.handbook" in settings.INSTALLED_APPS
