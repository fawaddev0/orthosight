"""Focused RAG pipeline tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from decision_tree.models import DiagnosisNode
from rag.embedding_generator import (
    ACTIVE_ALIAS,
    VECTOR_SIZE,
    IndexedSection,
    activate_collection,
    collection_name,
    index_sections,
    prepare_sections,
)
from rag.response_generator import AnswerRequest, QuestionnaireResponse, generate_plan
from rag.retriever import (
    CONDITION_QUERIES,
    RetrievedSection,
    _section_from_payload,
    retrieve,
)


def _write_sections(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _section(section_id: str, text: str, ready: bool = True) -> dict[str, object]:
    return {
        "book": "shoulder_book",
        "section_id": section_id,
        "heading_path": ["Shoulder", "Exercises"],
        "page_start": 3,
        "page_end": 4,
        "text": text,
        "ready_for_embedding": ready,
        "paragraphs": [],
    }


def test_prepare_sections_preserves_ready_section_without_splitting(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    text = "First complete sentence.\n\nSecond complete sentence."
    _write_sections(path, [_section("one", text), _section("two", "Flagged.", False)])

    records = prepare_sections(path, lambda value: len(value.split()), max_tokens=20)

    assert len(records) == 1
    assert records[0].section_id == "one"
    assert records[0].text == text
    assert records[0].heading_path == ["Shoulder", "Exercises"]
    assert records[0].page_start == 3
    assert records[0].page_end == 4


def test_prepare_sections_rejects_oversized_section_before_indexing(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    _write_sections(path, [_section("too-long", "one two three four five six")])

    with pytest.raises(ValueError, match="too-long"):
        prepare_sections(path, lambda value: len(value.split()), max_tokens=4)


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(range(len(text.split())))


class _EmbeddingModel:
    tokenizer = _Tokenizer()
    max_seq_length = 8192

    def encode(self, texts: list[str], *, normalize_embeddings: bool) -> np.ndarray:
        vectors = np.zeros((len(texts), VECTOR_SIZE), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


def test_index_and_activate_complete_section(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    original = "First sentence.\n\nSecond sentence."
    _write_sections(path, [_section("one", original)])
    client = QdrantClient(":memory:")

    assert index_sections(path, "v1", model=_EmbeddingModel(), client=client) == 1
    points, _ = client.scroll(collection_name("v1"), with_payload=True)
    assert len(points) == 1
    assert points[0].payload is not None
    assert points[0].payload["text"] == original

    with pytest.raises(ValueError, match="unexpected section count"):
        activate_collection("v1", 2, client=client)
    assert client.get_aliases().aliases == []

    activate_collection("v1", 1, client=client)
    assert client.get_aliases().aliases[0].alias_name == ACTIVE_ALIAS


def test_index_rejects_model_that_would_truncate_sections(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    _write_sections(path, [_section("one", "one two three four five")])
    client = QdrantClient(":memory:")
    model = _EmbeddingModel()
    model.max_seq_length = 4

    with pytest.raises(ValueError, match="4 embedding tokens"):
        index_sections(path, "v1", model=model, client=client)
    assert not client.collection_exists(collection_name("v1"))


class _QueryEmbedder:
    def encode(self, text: str, *, normalize_embeddings: bool) -> np.ndarray:
        vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
        vector[0] = 1.0
        return vector


class _PairTokenizer:
    def encode(self, query: str, passage: str, *, add_special_tokens: bool, truncation: bool) -> list[int]:
        return list(range(len((query + " " + passage).split())))


class _Reranker:
    tokenizer = _PairTokenizer()
    max_length = 8192

    def __init__(self) -> None:
        self.pairs: list[list[str]] = []

    def predict(self, pairs: list[list[str]]) -> np.ndarray:
        self.pairs = pairs
        return np.array([float(pair[1].split("rank ")[-1]) for pair in pairs], dtype=np.float32)


def test_retrieve_reranks_full_sections_and_excludes_surgical_book() -> None:
    client = QdrantClient(":memory:")
    client.create_collection(ACTIVE_ALIAS, vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE))
    for rank in range(6):
        client.upsert(ACTIVE_ALIAS, points=[models.PointStruct(
            id=rank + 1,
            vector=[1.0] + [0.0] * (VECTOR_SIZE - 1),
            payload={
                "book": "shoulder_rotatory_cuff",
                "section_id": f"section-{rank}",
                "heading_path": ["Rotator cuff", "Exercise"],
                "text": f"Rotator cuff home exercise. Full sentence. rank {rank}",
                "page_start": 1,
                "page_end": 1,
            },
        )])
    client.upsert(ACTIVE_ALIAS, points=[models.PointStruct(
        id=20,
        vector=[1.0] + [0.0] * (VECTOR_SIZE - 1),
        payload={
            "book": "shoulder_surgical_stuff",
            "section_id": "surgery",
            "heading_path": ["Rotator cuff", "Surgery"],
            "text": "Rotator cuff postoperative exercise. rank 20",
            "page_start": 1,
            "page_end": 1,
        },
    )])
    reranker = _Reranker()

    hits = retrieve("4820", client=client, embedder=_QueryEmbedder(), reranker=reranker)

    assert [hit.section.section_id for hit in hits] == [
        "section-5", "section-4", "section-3", "section-2", "section-1", "section-0"
    ]
    assert all("Full sentence." in pair[1] for pair in reranker.pairs)
    assert all(hit.section.book != "shoulder_surgical_stuff" for hit in hits)


def test_section_payload_uses_complete_parent_text_and_stable_parent_id() -> None:
    section = _section_from_payload({
        "book": "Kisner",
        "text": "Pendulum exercises can be used for joint stretching.",
        "parent_text": "Adhesive capsulitis treatment. Pendulum exercises can be used for joint stretching.",
        "parent_id": "Kisner|p491|118",
        "child_idx": 2,
        "section_path": "Shoulder > Adhesive Capsulitis",
        "pdf_page": 491,
        "page_start": 491,
        "page_end": 491,
    })

    assert section.section_id == "Kisner|p491|118"
    assert section.heading_path == ["Shoulder", "Adhesive Capsulitis"]
    assert section.text.startswith("Adhesive capsulitis treatment.")


def test_retrieve_rejects_referral_diagnosis_before_search() -> None:
    with pytest.raises(ValueError, match="Unsupported diagnosis"):
        retrieve("4910")


def test_tendon_impingement_query_targets_conservative_exercise_evidence() -> None:
    query = CONDITION_QUERIES["4820"].casefold()

    assert "painful shoulder syndromes without dislocation" in query
    assert "pendulum exercises without weights" in query
    assert "rotator cuff" in query
    assert "postoperative" not in query


class _Generator:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = json.dumps(response)
        self.prompt = ""
        self.system = ""

    def count_tokens(self, system: str, user: str) -> int:
        return len((system + user).split())

    def generate(self, system: str, user: str) -> str:
        self.system = system
        self.prompt = user
        return self.response


def _answer_request(diagnosis_id: str = "4820") -> AnswerRequest:
    diagnosis = DiagnosisNode(
        id=diagnosis_id,
        type="diagnosis",
        title="Rotator cuff tendinopathy",
        primaryDiagnosis="Conservative treatment",
        differentials=(),
        confidenceScore=75,
    )
    return AnswerRequest(
        diagnosis=diagnosis,
        patient_history="Prior right shoulder pain; no surgery.",
        questionnaire=[
            QuestionnaireResponse(question_id="q1", question="Pain duration?", answer="Six weeks"),
            QuestionnaireResponse(question_id="q2", question="Pain with lifting?", answer="Yes"),
        ],
    )


def _exercise_hit() -> RetrievedSection:
    return RetrievedSection(
        section=IndexedSection(
            book="shoulder_rotatory_cuff",
            section_id="exercise-1",
            heading_path=["Rotator cuff", "Exercise"],
            text="Pendulum exercise: move arm gently. Perform 3 sets of 10 repetitions.",
            page_start=8,
            page_end=8,
        ),
        score=0.9,
    )


def test_answer_uses_full_patient_input_and_verified_sets_and_reps() -> None:
    generator = _Generator({"exercises": [{
        "name": "Pendulum exercise",
        "sets": 3,
        "reps": 10,
        "instructions": "move arm gently",
        "source_section_id": "exercise-1",
        "evidence_quote": "Pendulum exercise: move arm gently. Perform 3 sets of 10 repetitions.",
    }]})
    request = _answer_request()

    plan = generate_plan(request, retrieve_sections=lambda condition: [_exercise_hit()], generator=generator)

    assert plan.status == "ready"
    assert [(exercise.sets, exercise.reps) for exercise in plan.exercises] == [(3, 10)]
    assert (plan.exercises[0].source_page_start, plan.exercises[0].source_page_end) == (8, 8)
    assert request.patient_history in generator.prompt
    assert all(response.answer in generator.prompt for response in request.questionnaire)
    assert request.diagnosis.id in generator.prompt


@pytest.mark.parametrize("sets,reps", [(3, 12), (3, None)])
def test_answer_rejects_unsupported_or_missing_dosage(sets: int, reps: int | None) -> None:
    generator = _Generator({"exercises": [{
        "name": "Pendulum exercise",
        "sets": sets,
        "reps": reps,
        "instructions": "move arm gently",
        "source_section_id": "exercise-1",
        "evidence_quote": "Pendulum exercise: move arm gently. Perform 3 sets of 10 repetitions.",
    }]})

    plan = generate_plan(_answer_request(), retrieve_sections=lambda condition: [_exercise_hit()], generator=generator)

    assert plan.status == "insufficient_evidence"
    assert plan.exercises == []


def test_answer_rejects_referral_diagnosis_before_retrieval() -> None:
    with pytest.raises(ValueError, match="Unsupported diagnosis"):
        generate_plan(_answer_request("4910"), retrieve_sections=lambda condition: [_exercise_hit()])


def test_answer_rejects_postoperative_exercise_for_conservative_diagnosis() -> None:
    hit = _exercise_hit()
    postoperative = RetrievedSection(
        section=hit.section.model_copy(update={
            "heading_path": ["Rotator cuff", "Postoperative exercise"],
            "text": "Pendulum exercise: move arm gently. Perform 3 sets of 10 repetitions.",
        }),
        score=hit.score,
    )
    generator = _Generator({"exercises": [{
        "name": "Pendulum exercise",
        "sets": 3,
        "reps": 10,
        "instructions": "move arm gently",
        "source_section_id": "exercise-1",
        "evidence_quote": "Pendulum exercise: move arm gently. Perform 3 sets of 10 repetitions.",
    }]})

    plan = generate_plan(_answer_request(), retrieve_sections=lambda condition: [postoperative], generator=generator)

    assert plan.status == "insufficient_evidence"
    assert plan.exercises == []
    assert generator.prompt == ""


def test_answer_rejects_evidence_for_other_condition() -> None:
    generator = _Generator({"exercises": [{
        "name": "Pendulum exercise",
        "sets": 3,
        "reps": 10,
        "instructions": "move arm gently",
        "source_section_id": "exercise-1",
        "evidence_quote": "Pendulum exercise: move arm gently. Perform 3 sets of 10 repetitions.",
    }]})

    plan = generate_plan(_answer_request("4810"), retrieve_sections=lambda condition: [_exercise_hit()], generator=generator)

    assert plan.status == "insufficient_evidence"
    assert generator.prompt == ""


def test_answer_accepts_source_supported_exercise_without_prescribed_dosage() -> None:
    hit = _exercise_hit()
    no_dosage = RetrievedSection(
        section=hit.section.model_copy(update={"text": "Rotator cuff pendulum exercise: move arm gently."}),
        score=hit.score,
    )
    generator = _Generator({"exercises": [{
        "name": "Pendulum exercise",
        "sets": None,
        "reps": None,
        "instructions": "",
        "source_section_id": "exercise-1",
        "evidence_quote": "Rotator cuff pendulum exercise: move arm gently.",
    }]})

    plan = generate_plan(_answer_request(), retrieve_sections=lambda condition: [no_dosage], generator=generator)

    assert plan.status == "ready"
    assert len(plan.exercises) == 1
    assert plan.exercises[0].sets is None
    assert plan.exercises[0].reps is None
    assert "Rotator cuff pendulum exercise" in generator.prompt
    assert "copy the exact exercise name" in generator.system.casefold()
    assert "return each distinct exercise" in generator.system.casefold()


def test_answer_accepts_exercise_name_with_plural_source_wording() -> None:
    quote = "Pendulum exercises without weights can cause gentle shoulder oscillation."
    generator = _Generator({"exercises": [{
        "name": "Pendulum exercise",
        "sets": None,
        "reps": None,
        "instructions": "",
        "source_section_id": "exercise-1",
        "evidence_quote": quote,
    }]})
    hit = RetrievedSection(
        section=_exercise_hit().section.model_copy(update={"text": quote}),
        score=0.9,
    )

    plan = generate_plan(
        _answer_request(),
        retrieve_sections=lambda condition: [hit],
        generator=generator,
    )

    assert plan.status == "ready"
    assert len(plan.exercises) == 1
    assert plan.exercises[0].name == "Pendulum exercise"


def test_answer_requotes_source_when_model_name_is_elsewhere_in_section() -> None:
    source_text = (
        "Pendulum exercises without weights are gentle. "
        "Multiple-angle muscle setting and protected stabilization exercises are initiated."
    )
    generator = _Generator({"exercises": [{
        "name": "Multiple-angle muscle setting",
        "sets": None,
        "reps": None,
        "instructions": "",
        "source_section_id": "exercise-1",
        "evidence_quote": "Pendulum exercises without weights are gentle.",
    }]})
    hit = RetrievedSection(
        section=_exercise_hit().section.model_copy(update={"text": source_text}),
        score=0.9,
    )

    plan = generate_plan(
        _answer_request(),
        retrieve_sections=lambda condition: [hit],
        generator=generator,
    )

    assert plan.status == "ready"
    assert plan.exercises[0].name == "Multiple-angle muscle setting"
    assert "Multiple-angle muscle setting" in plan.exercises[0].evidence_quote
    assert plan.exercises[0].evidence_quote in source_text


def test_answer_deduplicates_pendulum_aliases() -> None:
    first = _exercise_hit()
    second = RetrievedSection(
        section=first.section.model_copy(update={
            "section_id": "exercise-2",
            "text": "Rotator cuff Pendulum (Codman's) exercises use gravity to distract the humerus.",
        }),
        score=0.8,
    )
    generator = _Generator({"exercises": [
        {
            "name": "Pendulum exercises",
            "sets": None,
            "reps": None,
            "instructions": "",
            "source_section_id": "exercise-1",
            "evidence_quote": first.section.text,
        },
        {
            "name": "Pendulum (Codman's) exercises",
            "sets": None,
            "reps": None,
            "instructions": "",
            "source_section_id": "exercise-2",
            "evidence_quote": second.section.text,
        },
    ]})

    plan = generate_plan(
        _answer_request(),
        retrieve_sections=lambda condition: [first, second],
        generator=generator,
    )

    assert plan.status == "ready"
    assert len(plan.exercises) == 1


def test_answer_keeps_source_supported_exercise_when_instructions_are_paraphrased() -> None:
    generator = _Generator({"exercises": [{
        "name": "Pendulum exercise",
        "sets": None,
        "reps": None,
        "instructions": "Gently swing the arm to improve mobility.",
        "source_section_id": "exercise-1",
        "evidence_quote": "Rotator cuff pendulum exercise: move arm gently.",
    }]})
    hit = RetrievedSection(
        section=_exercise_hit().section.model_copy(update={
            "text": "Rotator cuff pendulum exercise: move arm gently.",
        }),
        score=0.9,
    )

    plan = generate_plan(
        _answer_request(),
        retrieve_sections=lambda condition: [hit],
        generator=generator,
    )

    assert plan.status == "ready"
    assert len(plan.exercises) == 1
    assert plan.exercises[0].instructions == ""
    assert plan.exercises[0].evidence_quote == "Rotator cuff pendulum exercise: move arm gently."


def test_answer_prioritizes_home_exercise_evidence_over_background() -> None:
    background_hits = [
        RetrievedSection(
            section=_exercise_hit().section.model_copy(update={
                "section_id": f"background-{index}",
                "heading_path": ["Shoulder", "Adhesive Capsulitis", "Overview"],
                "text": "Adhesive capsulitis causes progressive shoulder stiffness.",
            }),
            score=1.0 - index / 10,
        )
        for index in range(4)
    ]
    exercise_hit = RetrievedSection(
        section=_exercise_hit().section.model_copy(update={
            "section_id": "home-exercise",
            "heading_path": ["Shoulder", "Adhesive Capsulitis", "Exercises"],
            "text": "Adhesive capsulitis: pendulum exercise gently stretches the joint.",
        }),
        score=0.2,
    )
    generator = _Generator({"exercises": []})

    generate_plan(
        _answer_request("4810"),
        retrieve_sections=lambda condition: [*background_hits, exercise_hit],
        generator=generator,
    )

    evidence = json.loads(generator.prompt)["evidence"]
    assert evidence[0]["section_id"] == "home-exercise"
