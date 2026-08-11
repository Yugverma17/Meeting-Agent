from __future__ import annotations

from quorum.llm.cache import LLMCache


def test_roundtrip(cache):
    key = LLMCache.make_key(model="m", prompt="hello")
    assert cache.get(key) is None
    cache.set(key, {"text": "world", "total_tokens": 7})
    assert cache.get(key)["text"] == "world"
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_key_is_order_independent_but_value_sensitive():
    a = LLMCache.make_key(model="m", prompt="p", temperature=0.0)
    b = LLMCache.make_key(temperature=0.0, prompt="p", model="m")
    assert a == b, "key must not depend on kwarg ordering"

    c = LLMCache.make_key(model="m", prompt="p", temperature=0.7)
    assert a != c, "changing a semantic input must miss the cache"


def test_prompt_edit_misses_cache():
    """Editing a prompt must not silently serve the previous answer."""
    before = LLMCache.make_key(model="m", prompt="Extract commitments.")
    after = LLMCache.make_key(model="m", prompt="Extract firm commitments only.")
    assert before != after


def test_disabled_cache_never_reads_or_writes(tmp_path):
    cache = LLMCache(tmp_path / "c", enabled=False)
    key = LLMCache.make_key(model="m", prompt="p")
    cache.set(key, {"text": "x"})
    assert cache.get(key) is None
    assert cache.size() == (0, 0)


def test_corrupt_entry_is_treated_as_miss_and_removed(cache):
    key = LLMCache.make_key(model="m", prompt="p")
    cache.set(key, {"text": "ok"})
    path = cache._path(key)
    path.write_text("{{{ broken", encoding="utf-8")

    assert cache.get(key) is None
    assert not path.exists(), "a corrupt entry should be dropped, not left to fail again"
    assert cache.stats.errors == 1


def test_hit_rate_and_size(cache):
    for i in range(3):
        cache.set(LLMCache.make_key(model="m", prompt=str(i)), {"text": str(i)})
    for i in range(3):
        cache.get(LLMCache.make_key(model="m", prompt=str(i)))

    count, total_bytes = cache.size()
    assert count == 3
    assert total_bytes > 0
    assert cache.stats.hit_rate == 1.0


def test_clear_removes_everything(cache):
    cache.set(LLMCache.make_key(model="m", prompt="p"), {"text": "x"})
    assert cache.clear() == 1
    assert cache.size()[0] == 0
