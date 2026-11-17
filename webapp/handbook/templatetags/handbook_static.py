"""{% static_v "handbook/app.js" %}: a static URL with a fingerprint of the file's contents.

When a file changes, its URL changes, so browsers fetch the new version instead of reusing a
cached old one. The fingerprint is recomputed only when the file's modification time changes.
"""

import hashlib

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()
_cache: dict[str, tuple[float, str]] = {}


@register.simple_tag
def static_v(path: str) -> str:
    url = static(path)
    found = finders.find(path)
    if not found:
        return url
    try:
        import os
        mtime = os.path.getmtime(found)
        cached = _cache.get(path)
        if cached is None or cached[0] != mtime:
            with open(found, "rb") as f:
                _cache[path] = (mtime, hashlib.sha256(f.read()).hexdigest()[:10])
        return f"{url}?v={_cache[path][1]}"
    except OSError:
        return url
