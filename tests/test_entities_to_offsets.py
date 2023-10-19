import pytest
from camembert_bio.utils.offsets_evaluation import tags_to_entities_with_offsets  # Adjust the import to your file structure

@pytest.mark.parametrize(
    "tokens,tags,text,expected",
    [
        (
            ["Hawking", "was", "a", "theoretical", "physicist"],
            ["B-PER", "O", "O", "O", "O"],
            "Hawking was a theoretical physicist.",
            [{'type': 'PER', 'offsets': [(0, 7)]}],
        ),
        (
            ["Steve", "Jobs", "and", "Steve", "Wozniak", "founded", "Apple"],
            ["B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "B-ORG"],
            "Steve Jobs and Steve Wozniak founded Apple.",
            [
                {'type': 'PER', 'offsets': [(0, 5), (6, 10)]},
                {'type': 'PER', 'offsets': [(15, 20), (21, 28)]},
                {'type': 'ORG', 'offsets': [(37, 42)]}
            ],
        ),
        (
            ["New", "York", "is", "a", "city", "in", "the", "USA"],
            ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC"],
            "New York is a city in the USA.",
            [
                {'type': 'LOC', 'offsets': [(0, 3), (4, 8)]},
                {'type': 'LOC', 'offsets': [(26, 29)]}
            ],
        ),
        (
            ["She", "lives", "in", "San", "Francisco"],
            ["O", "O", "O", "B-LOC", "I-LOC"],
            "She lives in San Francisco.",
            [{'type': 'LOC', 'offsets': [(13, 16), (17, 26)]}],
        )
    ]
)
def test_tags_to_entities_with_offsets(tokens, tags, text, expected):
    result = tags_to_entities_with_offsets(tokens, tags, text)
    assert result == expected
