import pytest

from camembert_bio.data_handling.nested_ner_dataset import NestedPerDepthNERPreprocessor

def test_label_mapping():
    # Sample data for testing
    data = {
        "train": [{
            "entities": [
                {"type": "CHEM", "offsets": [(0,4)]},
                {"type": "LIVB", "offsets": [(5,9)]}
            ]
        }]
    }

    
    preprocessor = NestedPerDepthNERPreprocessor(data, "fr")
    
    assert "B-CHEM" in preprocessor.label2id
    assert "I-LIVB" in preprocessor.label2id
    assert preprocessor.label2id["O"] == 0

def test_compute_depth():
    # Sample data for testing
    entities = [
        {"offsets": [(10, 20)]},
        {"offsets": [(15, 20)]},
        {"offsets": [(5, 12), (30, 40)]}  # Discontinuous entity
    ]

    preprocessor = NestedPerDepthNERPreprocessor({}, "fr")
    depths = preprocessor._compute_depth(entities)
    
    assert depths[(10, 20)] == 1
    assert depths[(15, 20)] == 2
    assert (30, 40) not in depths

def test_process_data():
    # Sample data for testing
    data = {
        "train": [{
            "passages": [{
                "text": ["This is a test sentence."]
            }],
            "entities": [
                {"type": "CHEM", "text": ["test"], "offsets": [(10, 14)]}
            ]
        }]
    }
    
    preprocessor = NestedPerDepthNERPreprocessor(data, "fr")
    processed = preprocessor.process_data("train")
    
    # Based on the data, there should be labels for the word "test" in the sentence
    assert processed[0]['tokens'][3] == "test"
    assert processed[0]['ner_tags_layer1'][3] == preprocessor.label2id["B-CHEM"]


def test_nested_entities():
    data = {
        "train": [{
            "passages": [{
                "text": ["This is a nested entity test."]
            }],
            "entities": [
                {"type": "CHEM", "text": ["nested entity"], "offsets": [(10, 23)]},
                {"type": "LIVB", "text": ["entity"], "offsets": [(17, 23)]}
            ]
        }]
    }
    
    preprocessor = NestedPerDepthNERPreprocessor(data, "fr")
    processed = preprocessor.process_data("train")
    
    assert processed[0]['tokens'][3:5] == ["nested", "entity"]
    assert processed[0]['ner_tags_layer1'][3] == preprocessor.label2id["B-CHEM"]
    assert processed[0]['ner_tags_layer1'][4] == preprocessor.label2id["I-CHEM"]
    assert processed[0]['ner_tags_layer2'][4] == preprocessor.label2id["B-LIVB"]

def test_multiple_sentences():
    data = {
        "train": [{
            "passages": [{
                "text": ["This is the first sentence. This is the second sentence."]
            }],
            "entities": [
                {"type": "CHEM", "text": ["first"], "offsets": [(12, 17)]},
                {"type": "LIVB", "text": ["second"], "offsets": [(40, 46)]}
            ]
        }]
    }
    
    preprocessor = NestedPerDepthNERPreprocessor(data, "fr")
    processed = preprocessor.process_data("train")
    
    # Entity from the first sentence
    assert processed[0]['tokens'][3] == "first"
    assert processed[0]['ner_tags_layer1'][3] == preprocessor.label2id["B-CHEM"]
    
    print(processed[1])
    # Entity from the second sentence
    assert processed[1]['tokens'][3] == "second"
    assert processed[1]['ner_tags_layer1'][3] == preprocessor.label2id["B-LIVB"]

def test_entity_at_start_and_end():
    data = {
        "train": [{
            "passages": [{
                "text": ["Entity at the start and entity at the end."]
            }],
            "entities": [
                {"type": "CHEM", "text": ["Entity"], "offsets": [(0, 6)]},
                {"type": "LIVB", "text": ["end"], "offsets": [(38, 41)]}
            ]
        }]
    }
    
    preprocessor = NestedPerDepthNERPreprocessor(data, "fr")
    processed = preprocessor.process_data("train")
    
    # Entity at the start
    assert processed[0]['tokens'][0] == "Entity"
    assert processed[0]['ner_tags_layer1'][0] == preprocessor.label2id["B-CHEM"]
    
    # Entity at the end
    assert processed[0]['tokens'][-2] == "end"
    assert processed[0]['ner_tags_layer1'][-2] == preprocessor.label2id["B-LIVB"]



    