"""Chunking of the synthetic maintenance notes, offline."""
from search.build_index import NOTES, chunk_note, load_chunks, parse_note


def test_every_note_is_parsed_with_its_failure_type():
    notes = [parse_note(path) for path in sorted(NOTES.glob("*.md")) if path.name != "README.md"]
    assert len(notes) == 20
    assert {note["failure_type"] for note in notes} == {"TWF", "HDF", "PWF", "OSF", "RNF", "GENERAL"}
    assert all("Synthetic note" in note["body"] for note in notes)


def test_each_note_gives_one_chunk_per_section():
    chunks = load_chunks()
    assert len(chunks) == 40
    assert {chunk["section"] for chunk in chunks} == {"When to use", "Procedure"}
    assert len({chunk["id"] for chunk in chunks}) == 40


def test_chunks_carry_title_and_type_and_skip_the_banner():
    note = parse_note(NOTES / "hdf-01-cooling-inspection.md")
    chunks = chunk_note(note)
    assert [c["id"] for c in chunks] == ["hdf-01-cooling-inspection-1", "hdf-01-cooling-inspection-2"]
    assert all(c["failure_type"] == "HDF" and c["title"] == "Cooling system inspection" for c in chunks)
    assert chunks[0]["embedding_text"].startswith("Cooling system inspection. When to use: ")
    assert "8.6 K" in chunks[0]["content"]
    assert all("Synthetic note" not in c["content"] for c in chunks)
