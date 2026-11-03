"""Assign ordered IDs and linear links to the etiology question spec."""

import argparse
import json
import os
import tempfile
from pathlib import Path

TEXT_FIELDS = ("verbatimText", "questionText", "sourceDoc", "clinicalRelevance")


def link_spec(path: Path) -> None:
    spec = json.loads(path.read_text(encoding="utf-8"))
    questions = spec["etiology_questions"]
    original_text = [tuple(question[field] for field in TEXT_FIELDS) for question in questions]

    for index, question in enumerate(questions, start=1):
        next_node_id = (
            f"etiology_q{index + 1:02d}"
            if index < len(questions)
            else "etiology_module_complete"
        )
        question["id"] = f"etiology_q{index:02d}"
        if not question["options"]:
            question["options"] = [
                {"value": "continue", "label": "Continue", "nextNodeId": next_node_id}
            ]
        else:
            for option in question["options"]:
                option["nextNodeId"] = next_node_id

    assert original_text == [
        tuple(question[field] for field in TEXT_FIELDS) for question in questions
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        json.dump(spec, output, ensure_ascii=False, indent=2)
        output.write("\n")
        temporary_path = Path(output.name)
    os.replace(temporary_path, path)


def main() -> None:
    default_spec = Path(__file__).resolve().parents[1] / "docs/specs/01_etiology_spec.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=default_spec)
    args = parser.parse_args()
    link_spec(args.spec)


if __name__ == "__main__":
    main()
