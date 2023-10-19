from datasets import load_dataset
from data_handling.nested_ner_dataset import NestedPerDepthNERPreprocessor, NestedNERDataModule 
from models.nested_ner_bert import NestedPerDepthNERModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from camembert_bio.evaluation.evaluate_offsets import EvaluationCallback, evaluate_model

def main():
    wandb_logger = WandbLogger(log_model="all")

    data = load_dataset("bigbio/quaero", name="quaero_medline_bigbio_kb")

    preprocessor = NestedPerDepthNERPreprocessor(data, "fr")

    data_train = preprocessor.process_data("train")
    data_val = preprocessor.process_data("validation")

    data_module = NestedNERDataModule(data_train, data_val, batch_size=16, max_length=512, num_workers=8)

    model = NestedPerDepthNERModel(
        n_depth=preprocessor.n_layers, id2label=preprocessor.id2label, stack_depths=True, learning_rate=1e-5, dropout_prob=0
    )

    evaluation_callback = EvaluationCallback(test_data=data["test"])

    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        devices=1,
        precision="16",
        fast_dev_run=False,
        callbacks=[evaluation_callback]
    )

    trainer.fit(model, data_module)
    results = evaluate_model(model, data["test"])
    print(results)
    wandb_logger.log_metrics(results)

if __name__ == "__main__":
    main()