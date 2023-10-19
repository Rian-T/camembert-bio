import logging
from pytorch_lightning.callbacks import Callback

# Configure logging
logging.basicConfig(level=logging.INFO)

class EvaluationCallback(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data

    def on_train_epoch_end(self, trainer, pl_module):
        print("on_epoch_end called")
        logging.info(f'Evaluating model at epoch {trainer.current_epoch}')
        evaluation_results = evaluate_model(pl_module, self.test_data)
        trainer.logger.log_metrics(evaluation_results, step=trainer.global_step)
        logging.info(f'Evaluation results: {evaluation_results}')

def compare_entities(predicted_entities, true_entities):
    exact_matches = 0
    for predicted_entity in predicted_entities:
        for true_entity in true_entities:
            if predicted_entity['offsets'] == true_entity['offsets'] and predicted_entity['type'] == true_entity['type']:
                exact_matches += 1
    return exact_matches, len(predicted_entities), len(true_entities)


def evaluate_model(model, test_data):
    total_exact_matches = 0
    total_predicted_entities = 0
    total_true_entities = 0
    for idx, example in enumerate(test_data):
        predicted_entities = model.predict_example(example)
        true_entities = example['entities']
        exact_matches, predicted_entities_count, true_entities_count = compare_entities(predicted_entities, true_entities)
        total_exact_matches += exact_matches
        total_predicted_entities += predicted_entities_count
        total_true_entities += true_entities_count
    precision = total_exact_matches / total_predicted_entities if total_predicted_entities > 0 else 0
    recall = total_exact_matches / total_true_entities if total_true_entities > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    logging.info(f'Total exact matches: {total_exact_matches}, Total predicted entities: {total_predicted_entities}, Total true entities: {total_true_entities}')
    return {
        'offsets/test_precision': precision,
        'offsets/test_recall': recall,
        'offsets/test_f1': f1
    }