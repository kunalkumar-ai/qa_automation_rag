import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    parent_id: str
    section_name: str
    chunk_type: str  # "parent" or "child"


def parse_sections(text: str) -> list[dict]:
    """Find every ITEM heading in the document and split text at those boundaries.

    Returns a list of {name, text} dicts — one per section.
    """
    pattern = re.compile(r'(ITEM\s+\d+[A-C]?\.\s+[A-Z][A-Z\s,&\(\)]+)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({"name": name, "text": text[start:end].strip()})
    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Split section text at blank lines, dropping paragraphs shorter than 10 words."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if len(p.strip().split()) >= 10]


def build_chunks(text: str) -> list[Chunk]:
    """Build the parent/child chunk hierarchy from the full document text.

    Each section becomes one parent chunk. Each paragraph within a section
    becomes one child chunk that stores its parent_id for later lookup.
    """
    sections = parse_sections(text)
    all_chunks: list[Chunk] = []

    for i, section in enumerate(sections):
        parent_id = f"parent_{i}"

        all_chunks.append(Chunk(
            chunk_id=parent_id,
            text=section["text"],
            parent_id=parent_id,
            section_name=section["name"],
            chunk_type="parent",
        ))

        for j, para in enumerate(_split_paragraphs(section["text"])):
            all_chunks.append(Chunk(
                chunk_id=f"child_{i}_{j}",
                text=para,
                parent_id=parent_id,
                section_name=section["name"],
                chunk_type="child",
            ))

    return all_chunks
