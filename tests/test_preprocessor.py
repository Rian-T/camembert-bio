import pytest
from camembert_bio.data_handling.ner_dataset import NERPreprocessor

# Define a fixture that creates a NERPreprocessor instance with different granularities
@pytest.fixture(params=["smaller-preference", "bigger-preference"])
def preprocessor(request):
    data = {
        "train": [
            {
                "passages": [{"text": ["This is a test sentence."]}],
                "entities": [
                    {"type": "test", "offsets": [(10, 14)]},
                    {"type": "test-sentence", "offsets": [(10, 22)]},
                    {"type": "sentence", "offsets": [(15, 22)]}
                ]
            }
        ]
    }
    return NERPreprocessor(data, lang="en", granularity=request.param)

# Helper function to check label mappings
def check_label_mappings(label2id, id2label):
    assert label2id["O"] == 0
    assert id2label[0] == "O"
    assert label2id["B-sentence"] == 1
    assert id2label[1] == "B-sentence"
    assert label2id["B-test"] == 2
    assert id2label[2] == "B-test"
    assert label2id["B-test-sentence"] == 3
    assert id2label[3] == "B-test-sentence"
    assert label2id["I-sentence"] == 4
    assert id2label[4] == "I-sentence"
    assert label2id["I-test"] == 5
    assert id2label[5] == "I-test"
    assert label2id["I-test-sentence"] == 6
    assert id2label[6] == "I-test-sentence" 

# Test that the _label_mapping method correctly creates label mappings
def test_label_mapping(preprocessor):
    label2id, id2label = preprocessor._label_mapping()
    check_label_mappings(label2id, id2label)

# Test that the process_data method correctly processes the data
def test_process_data(preprocessor):
    processed_data = preprocessor.process_data("train")
    label2id, _ = preprocessor._label_mapping()
    assert len(processed_data) == 1
    assert processed_data[0]["tokens"] == ["This", "is", "a", "test", "sentence", "."]
    if preprocessor.granularity == "smaller-preference":
        assert processed_data[0]["ner_tags"] == [0, 0, 0, label2id["B-test"], label2id["B-sentence"], 0]
    else:
        assert processed_data[0]["ner_tags"] == [0, 0, 0, label2id["B-test-sentence"], label2id["I-test-sentence"], 0]

# Test that the process_data method correctly handles overlapping entities
def test_overlapping_entities(preprocessor):
    label2id, _ = preprocessor._label_mapping()
    processed_data = preprocessor.process_data("train")
    assert len(processed_data) == 1
    assert processed_data[0]["tokens"] == ["This", "is", "a", "test", "sentence", "."]
    if preprocessor.granularity == "smaller-preference":
        assert processed_data[0]["ner_tags"] == [0, 0, 0, label2id["B-test"], label2id["B-sentence"], 0]  # "test" entity is preferred
    else:
        assert processed_data[0]["ner_tags"] == [0, 0, 0, label2id["B-test-sentence"], label2id["I-test-sentence"], 0]  # "test2" entity is preferred