"""Integration checks against the current shoulder PDF excerpts."""

import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pymupdf
import pytest
from pydantic import BaseModel, Field

from rag.book_preprocessor import Config, join_lines, process_book

BOOKS = Path(__file__).resolve().parents[1] / "src" / "rag" / "test_books"
PAGE_COUNTS = {
    "shoulder_joint_pathologies": 21,
    "shoulder_joint_structure": 14,
    "shoulder_rotatory_cuff": 16,
    "shoulder_surgical_stuff": 16,
}
PDF_SENTENCES = {
    "shoulder_joint_pathologies": (
        "For home exercise, emphasize “curling” the head and neck, not lifting the head up.",
        "Have the patient practice tucking the chin and curling the head up.",
    ),
    "shoulder_joint_structure": (
        "Begin training awareness of safe spinal motions and the neutral spine position or bias.",
        "Have patient learn to activate the deep (core) stabilizing musculature while in the neutral position.",
    ),
    "shoulder_rotatory_cuff": (
        "The combined actions of the deltoid and short rotators result in a balance of forces that abduct the humerus and control the humeral head.",
        "Common sites for compression are the scalene triangle, costoclavicular space and under the coracoid process, and pectoralis minor muscle.",
    ),
    "shoulder_surgical_stuff": (
        "A decrease in pain is the most dramatic result of glenohumeral arthroplasty.",
        "After the immobilization device may be removed for exercise, begin active scapulothoracic ROM.",
    ),
}


class Source(BaseModel):
    source_id: str
    page: int
    bbox: tuple[float, float, float, float]
    column: str
    region: str


class Paragraph(BaseModel):
    text: str
    sources: list[Source]
    join_evidence: list[str]
    quality_flags: list[str]


class Section(BaseModel):
    heading_path: list[str]
    page_start: int
    page_end: int
    paragraphs: list[Paragraph]
    text: str
    n_chars: int
    ready_for_embedding: bool


class Table(BaseModel):
    page: int
    rows: list[list[str | None]]


class DroppedFragment(BaseModel):
    paragraph_id: str
    source_ids: list[str]
    text: str


class Report(BaseModel):
    pages: int
    sections_kept: int
    tables_extracted: int
    chars_kept: int
    paragraphs_total: int
    paragraphs_ending_mid_sentence: int
    review_items: int
    quality_status: str
    incomplete_fragments_dropped: list[DroppedFragment] = Field(default_factory=list)


@dataclass(frozen=True)
class BookOutput:
    path: Path
    sections: tuple[Section, ...]
    tables: tuple[Table, ...]
    review_lines: tuple[str, ...]
    report: Report

    @property
    def text(self) -> str:
        return "\n\n".join(section.text for section in self.sections)

    @property
    def paragraphs(self) -> tuple[Paragraph, ...]:
        return tuple(paragraph for section in self.sections for paragraph in section.paragraphs)


@pytest.fixture(scope="module")
def processed_books(tmp_path_factory: pytest.TempPathFactory) -> dict[str, BookOutput]:
    paths = sorted(BOOKS.glob("*.pdf"))
    assert {path.stem for path in paths} == set(PAGE_COUNTS), "PDF fixture set changed"
    hashes_before = {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}
    output_root = tmp_path_factory.mktemp("book_preprocessing")
    outputs: dict[str, BookOutput] = {}

    for path in paths:
        process_book(path, output_root, Config(ocr=False, keep_tables_separately=True))
        directory = output_root / path.stem
        sections = tuple(
            Section.model_validate_json(line)
            for line in (directory / "sections.jsonl").read_text(encoding="utf-8").splitlines()
        )
        tables = tuple(
            Table.model_validate_json(line)
            for line in (directory / "tables.jsonl").read_text(encoding="utf-8").splitlines()
        )
        review_lines = tuple((directory / "review.jsonl").read_text(encoding="utf-8").splitlines())
        report = Report.model_validate_json((directory / "report.json").read_text(encoding="utf-8"))
        outputs[path.stem] = BookOutput(path, sections, tables, review_lines, report)

    assert hashes_before == {path.name: sha256(path.read_bytes()).hexdigest() for path in paths}
    return outputs


@pytest.mark.parametrize("name", PAGE_COUNTS)
def test_pdf_outputs_are_consistent(processed_books: dict[str, BookOutput], name: str) -> None:
    output = processed_books[name]
    with pymupdf.open(output.path) as document:  # type: ignore[no-untyped-call]
        assert len(document) == PAGE_COUNTS[name]
    assert output.report.pages == PAGE_COUNTS[name]
    assert output.sections
    assert output.report.sections_kept == len(output.sections)
    assert output.report.tables_extracted == len(output.tables)
    assert output.report.chars_kept == sum(section.n_chars for section in output.sections)
    assert output.report.paragraphs_total == len(output.paragraphs)
    assert output.report.review_items == len(output.review_lines)
    assert output.report.quality_status in {"checks_passed", "needs_review"}
    for section in output.sections:
        assert section.text
        assert section.n_chars == len(section.text)
        assert 1 <= section.page_start <= section.page_end <= output.report.pages
        assert section.text.split("\n\n") == [paragraph.text for paragraph in section.paragraphs]
    for table in output.tables:
        assert 1 <= table.page <= output.report.pages
        assert table.rows


@pytest.mark.parametrize("name", PAGE_COUNTS)
def test_known_pdf_sentences_survive_once(processed_books: dict[str, BookOutput], name: str) -> None:
    output = processed_books[name]
    with pymupdf.open(output.path) as document:  # type: ignore[no-untyped-call]
        pdf_text = re.sub(r"\s+", " ", " ".join(page.get_text() for page in document))
    output_text = re.sub(r"\s+", " ", output.text)
    for sentence in PDF_SENTENCES[name]:
        assert pdf_text.count(sentence) == 1, (name, sentence, "PDF ground truth changed")
        assert output_text.count(sentence) == 1, (name, sentence)


@pytest.mark.parametrize("name", PAGE_COUNTS)
def test_paragraph_provenance_matches_pdf(processed_books: dict[str, BookOutput], name: str) -> None:
    output = processed_books[name]
    with pymupdf.open(output.path) as document:  # type: ignore[no-untyped-call]
        for section in output.sections:
            for paragraph in section.paragraphs:
                assert paragraph.sources, (name, paragraph.text)
                pdf_regions: list[str] = []
                for source in paragraph.sources:
                    assert section.page_start <= source.page <= section.page_end
                    assert source.source_id.startswith(f"p{source.page}-")
                    page = document[source.page - 1]
                    rect = pymupdf.Rect(source.bbox)  # type: ignore[no-untyped-call]
                    assert 0 <= rect.x0 < rect.x1 <= page.rect.width + 0.1
                    assert 0 <= rect.y0 < rect.y1 <= page.rect.height + 0.1
                    pdf_regions.append(page.get_text("text", clip=rect))
                output_words = set(re.findall(r"[a-z]{4,}", paragraph.text.casefold()))
                source_words = set(re.findall(r"[a-z]{4,}", " ".join(pdf_regions).casefold()))
                if len(output_words) >= 5:
                    overlap = len(output_words & source_words) / len(output_words)
                    assert overlap >= 0.75, (name, paragraph.text, pdf_regions, overlap)
                elif output_words:
                    assert output_words & source_words, (name, paragraph.text, pdf_regions)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["shoul-", "der motion."], "shoulder motion."),
        (["well-", "being matters."], "well-being matters."),
        (["A complete", "sentence."], "A complete sentence."),
    ],
)
def test_line_join_preserves_words(lines: list[str], expected: str) -> None:
    assert join_lines(lines) == expected


@pytest.mark.parametrize("name", PAGE_COUNTS)
def test_no_paragraph_ends_in_a_split_word(processed_books: dict[str, BookOutput], name: str) -> None:
    broken = [
        (paragraph.sources[0].page, paragraph.text[-100:])
        for paragraph in processed_books[name].paragraphs
        if re.search(r"[A-Za-z]-$", paragraph.text)
    ]
    assert not broken, (name, broken)


@pytest.mark.parametrize("name", PAGE_COUNTS)
def test_no_prose_ends_at_a_sentence_connector(
    processed_books: dict[str, BookOutput], name: str
) -> None:
    broken = [
        (paragraph.sources[0].page, paragraph.text[-120:])
        for paragraph in processed_books[name].paragraphs
        if len(paragraph.text.split()) >= 12
        and re.search(r"\b(?:and|or|the|of|to|with|for|in|that|this|a|an)$", paragraph.text, re.IGNORECASE)
    ]
    assert not broken, (name, broken)


@pytest.mark.parametrize(
    ("name", "phrase", "pages"),
    [
        ("shoulder_joint_pathologies", "to stabilize and control motion are emphasized", {1, 2}),
        (
            "shoulder_joint_pathologies",
            "the long head of the biceps and the long head of the triceps brachii reinforce",
            {18, 19},
        ),
        ("shoulder_rotatory_cuff", "repair of the rotator cuff is weaned", {12, 13}),
        ("shoulder_surgical_stuff", "for evaluation and an okay to advance", {14, 15}),
    ],
)
def test_page_spanning_prose_is_rejoined(
    processed_books: dict[str, BookOutput], name: str, phrase: str, pages: set[int]
) -> None:
    matches = [paragraph for paragraph in processed_books[name].paragraphs if phrase in paragraph.text]
    assert len(matches) == 1, (name, phrase)
    assert {source.page for source in matches[0].sources} == pages
    assert "cross_page_continuation" in matches[0].join_evidence


@pytest.mark.parametrize(
    ("name", "ending"),
    [
        ("shoulder_joint_structure", "stabilize and"),
        ("shoulder_rotatory_cuff", "permis-"),
    ],
)
def test_missing_pdf_continuations_are_reported(
    processed_books: dict[str, BookOutput], name: str, ending: str
) -> None:
    output = processed_books[name]
    assert len(output.report.incomplete_fragments_dropped) == 1
    assert output.report.incomplete_fragments_dropped[0].text.endswith(ending)
    assert output.report.incomplete_fragments_dropped[0].source_ids
    assert ending not in output.text
    assert output.report.quality_status == "needs_review"


@pytest.mark.parametrize(
    ("name", "page", "source_phrase"),
    [
        (
            "shoulder_joint_pathologies",
            2,
            (
                "Include pushing, pulling, reaching, and lifting "
                "(see Functional Training section later in this chapter)."
            ),
        ),
        ("shoulder_joint_structure", 7, "In addition, use of resistance loads"),
    ],
)
def test_source_prose_stays_in_one_paragraph(
    processed_books: dict[str, BookOutput], name: str, page: int, source_phrase: str
) -> None:
    output = processed_books[name]
    with pymupdf.open(output.path) as document:  # type: ignore[no-untyped-call]
        source_text = re.sub(r"\s+", " ", document[page - 1].get_text())
    assert source_phrase in source_text, (name, page, "PDF ground truth changed")
    matches = [paragraph for paragraph in output.paragraphs if source_phrase in paragraph.text]
    assert len(matches) == 1, (name, page, source_phrase)


def test_page_spanning_word_is_not_interrupted_by_box_label(
    processed_books: dict[str, BookOutput],
) -> None:
    output = processed_books["shoulder_joint_pathologies"]
    with pymupdf.open(output.path) as document:  # type: ignore[no-untyped-call]
        assert "physiologi-\n" in document[16].get_text()
        assert "cal motions)" in document[17].get_text()
    assert any("physiological motions" in paragraph.text for paragraph in output.paragraphs)
    assert all("physiologi- BOX 17.1" not in paragraph.text for paragraph in output.paragraphs)
