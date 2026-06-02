import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    text: str
    parent_id: str
    section_name: str
    chunk_type: str  # "parent" or "child"
    company: str = ""
    year: str = ""


# Minimum chars a sub-section must have to be saved as its own parent.
# Headings with no real content between them get merged into the next sub-section.
_MIN_SUBSECTION_CHARS = 150


def parse_sections(text: str) -> list[dict]:
    """Find every ITEM heading and split text at those boundaries."""
    # [^\n]+ captures the rest of the heading line only — avoids grabbing next line
    pattern = re.compile(r'(ITEM\s+\d+[A-C]?\.\s+[^\n]+)', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({"name": name, "text": text[start:end].strip()})
    return sections


def _is_subheading(line: str) -> bool:
    """Return True if line looks like an internal section sub-heading."""
    s = line.strip()
    if not s:
        return False
    words = s.split()
    return (
        1 <= len(words) <= 12 and
        len(s) <= 80 and
        s[0].isupper() and
        not s.isupper() and                        # exclude ALL CAPS lines
        not s.endswith('.') and
        not s.endswith(',') and
        not s.endswith(':') and
        '/' not in s and                           # exclude table rows with slashes
        '%' not in s and                           # exclude financial table rows
        not s.lower().startswith('total ') and     # exclude table subtotal rows
        not s[-1].isdigit() and                    # exclude lines ending in page numbers
        not re.search(r'\d{4}.*\d{4}', s)          # exclude table headers with two years
    )


def _split_subsections(section_name: str, section_text: str) -> list[dict]:
    """Split a section at internal sub-headings.

    Walks line by line. Any line that looks like a heading starts a new
    sub-section. Sub-sections with less than _MIN_SUBSECTION_CHARS of content
    are merged into the next sub-section to avoid useless tiny parents.
    If no sub-headings are found, returns the whole section as one entry.
    """
    subsections = []
    current_name = section_name
    current_lines: list[str] = []

    for line in section_text.split('\n'):
        if _is_subheading(line):
            text = '\n'.join(current_lines).strip()
            if len(text) >= _MIN_SUBSECTION_CHARS:
                subsections.append({"name": current_name, "text": text})
                current_name = f"{section_name} — {line.strip()}"
                current_lines = []
            else:
                # Too little content — keep accumulating under the same name
                # but update the heading to the new one
                current_name = f"{section_name} — {line.strip()}"
                current_lines = [l for l in current_lines if l.strip()]
        else:
            current_lines.append(line)

    text = '\n'.join(current_lines).strip()
    if text:
        subsections.append({"name": current_name, "text": text})

    return subsections if len(subsections) > 1 else [{"name": section_name, "text": section_text}]


_MAX_CHILD_CHARS = 6000  # ~1500 tokens — hard ceiling per child chunk


def _split_paragraphs(text: str) -> list[str]:
    """Split text at blank lines, then enforce a hard character ceiling.

    Paragraphs exceeding _MAX_CHILD_CHARS are split further at sentence
    boundaries so no child chunk ever exceeds the embedding model's token limit.
    Drops paragraphs shorter than 10 words.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    result = []
    for p in paragraphs:
        p = p.strip()
        if len(p.split()) < 10:
            continue
        if len(p) <= _MAX_CHILD_CHARS:
            result.append(p)
        else:
            # Split at sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current = ""
            for sentence in sentences:
                if len(current) + len(sentence) + 1 > _MAX_CHILD_CHARS and current:
                    result.append(current.strip())
                    current = sentence
                else:
                    current = current + " " + sentence if current else sentence
            if current.strip():
                result.append(current.strip())
    return result


def build_chunks(text: str, company: str = "", year: str = "") -> list[Chunk]:
    """Build the parent/child chunk hierarchy.

    Each sub-section becomes one parent chunk. Each paragraph within a
    sub-section becomes one child chunk storing its parent_id.
    company and year are tagged on every chunk for multi-document retrieval filtering.
    """
    sections = parse_sections(text)
    all_chunks: list[Chunk] = []
    prefix = f"{company}_{year}_" if company and year else ""

    for i, section in enumerate(sections):
        subsections = _split_subsections(section["name"], section["text"])

        for j, subsec in enumerate(subsections):
            parent_id = f"{prefix}parent_{i}_{j}"

            all_chunks.append(Chunk(
                chunk_id=parent_id,
                text=subsec["text"],
                parent_id=parent_id,
                section_name=subsec["name"],
                chunk_type="parent",
                company=company,
                year=year,
            ))

            for k, para in enumerate(_split_paragraphs(subsec["text"])):
                all_chunks.append(Chunk(
                    chunk_id=f"{prefix}child_{i}_{j}_{k}",
                    text=para,
                    parent_id=parent_id,
                    section_name=subsec["name"],
                    chunk_type="child",
                    company=company,
                    year=year,
                ))

    return all_chunks
