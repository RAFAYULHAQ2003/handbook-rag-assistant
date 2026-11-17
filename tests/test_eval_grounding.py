import pytest

from evals.matching import facts_matched
from tests.test_eval_set import ALL


@pytest.mark.parametrize("source,q", ALL, ids=[q["id"] for _, q in ALL])
def test_expected_facts_are_in_expected_documents(docs, source, q):
    text = "\n".join(docs[d].text for d in q["expected_doc_ids"])
    matched = facts_matched(text, q["must_include"])
    missing = [g for g, ok in zip(q["must_include"], matched, strict=True) if not ok]
    assert not missing, f"{q['id']}: not found in {q['expected_doc_ids']}: {missing}"
