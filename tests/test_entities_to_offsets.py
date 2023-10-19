from unittest.mock import MagicMock, patch
import pytest
import torch
from camembert_bio.utils.offsets_evaluation import tags_to_entities_with_offsets
from camembert_bio.models.nested_ner_bert import NestedPerDepthNERModel

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


# Mocked version of tags_to_entities_with_offsets function
def mock_tags_to_entities_with_offsets(tokens, tags, text):
    # Mocked behavior. Can be adjusted based on your testing needs.
    return [{'offsets': [(0, len(token))], 'token': token, 'tag': tag} for token, tag in zip(tokens, tags)]

@pytest.fixture
def mock_model():
    model = NestedPerDepthNERModel(n_depth=1, id2label={0: 'O', 1: 'Entity'})
    model.nlp = MagicMock(return_value=MagicMock(sentences=[MagicMock(tokens=['Sample', 'Text'], text='Sample Text')]))
    model.tokenizer.convert_tokens_to_ids = MagicMock(return_value=[0, 1])
    model.forward = MagicMock(return_value=[torch.tensor([[[0.7, 0.3], [0.5, 0.5]]])])  # Mock logits for 2 tokens
    model.logits_to_tags = MagicMock(return_value=[['O', 'Entity']])
    return model

def test_predict_example_functionality(mock_model):
    example = {'passages': [{'text': ['Sample Text']}]}

    with patch('camembert_bio.utils.offsets_evaluation.tags_to_entities_with_offsets', mock_tags_to_entities_with_offsets): 
        predicted_entities = mock_model.predict_example(example)

    assert len(predicted_entities) == 2  # Expecting two tokens in the output
    assert predicted_entities[0]['token'] == 'Sample'  # First token
    assert predicted_entities[0]['tag'] == 'O'  # Its corresponding tag
    assert predicted_entities[1]['token'] == 'Text'  # Second token
    assert predicted_entities[1]['tag'] == 'Entity'  # Its corresponding tag

