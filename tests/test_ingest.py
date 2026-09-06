from ingest import chunk_text, is_junk, looks_like_template


def test_chunk_text_overlaps_and_covers_text():
    text = "".join(str(i % 10) for i in range(2000))
    chunks = chunk_text(text, size=800, overlap=150)
    assert len(chunks) == 4                          # starts at 0, 650, 1300, 1950
    assert chunks[0] == text[:800]
    assert chunks[1][:150] == chunks[0][-150:]       # consecutive chunks overlap by 150
    assert all(len(c) <= 800 for c in chunks)


def test_chunk_text_short_input_gives_one_chunk():
    assert chunk_text("hello", size=800, overlap=150) == ["hello"]


def test_is_junk_flags_tiny_fragments():
    assert is_junk("too short")


def test_is_junk_keeps_normal_prose():
    prose = "Retrieval-augmented generation combines a retriever with a language model. " * 5
    assert not is_junk(prose)


def test_is_junk_flags_bibliography_entries():
    ref = ("[41] I. Mani, Proceedings of the Workshop on Learning from Imbalanced Datasets, "
           "2003. [42] A. Smith, Proceedings of the Conference on Language, pp. 1-10, 2019. "
           "Available: https://example.org [43] B. Jones, 2020.")
    assert is_junk(ref)


def test_looks_like_template_detects_few_shot_example():
    template = 'Example: Content: The capital is Paris. Question: What is it? Output: {"Reasoning": "x"}'
    assert looks_like_template(template)


def test_looks_like_template_ignores_normal_text():
    assert not looks_like_template("Rerankers rescore candidate passages using a cross-encoder.")