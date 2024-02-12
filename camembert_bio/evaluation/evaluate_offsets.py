import logging
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule

from tqdm import tqdm

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
    # Sort the entities by their offsets
    predicted_entities.sort(key=lambda e: e['offsets'])
    true_entities.sort(key=lambda e: e['offsets'])

    # Count the number of exact matches
    exact_matches = 0
    for pred in predicted_entities:
        if any(true['offsets'] == pred['offsets'] and true['type'] == pred['type'] for true in true_entities):
            exact_matches += 1

    return exact_matches, len(predicted_entities), len(true_entities)

def evaluate_model(model: LightningModule, test_data):
    """
    Evaluate the model's performance on the test data.

    Parameters:
    model (LightningModule): The model to evaluate.
    test_data (Dataset): The test data.

    Returns:
    dict: A dictionary with the precision, recall, and F1 score of the model.
    """
    # Initialize counters
    total_exact_matches = 0
    total_predicted_entities = 0
    total_true_entities = 0
    
    # Set the model to evaluation mode
    model.eval()
    
    # Process each batch
    for i in tqdm(range(0, len(test_data), 8)):
        examples = test_data[i:i+8]

        # Predict entities for each example in the batch
        predicted_entities_batch = model.predict_batch(examples)
        
        # Compare the predicted entities with the true entities
        for idx, entities in enumerate(examples["entities"]):
            predicted_entities = predicted_entities_batch[idx]
            true_entities = entities
            exact_matches, predicted_entities_count, true_entities_count = compare_entities(predicted_entities, true_entities)
            
            # Update counters
            total_exact_matches += exact_matches
            total_predicted_entities += predicted_entities_count
            total_true_entities += true_entities_count

    # Calculate metrics
    precision = total_exact_matches / total_predicted_entities if total_predicted_entities > 0 else 0
    recall = total_exact_matches / total_true_entities if total_true_entities > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    # Set the model back to training mode
    model.train()

    # Log the results
    logging.info(f'Total exact matches: {total_exact_matches}, Total predicted entities: {total_predicted_entities}, Total true entities: {total_true_entities}')
    
    # Return the results
    return {
        'offsets/test_precision': precision,
        'offsets/test_recall': recall,
        'offsets/test_f1': f1
    }