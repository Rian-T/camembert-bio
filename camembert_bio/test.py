from datasets import load_dataset
from data_handling.nested_ner_dataset import NestedPerDepthNERPreprocessor, NestedNERDataModule 
from models.nested_ner_bert import NestedPerDepthNERModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def main():
    wandb_logger = WandbLogger(log_model="all")

    data = load_dataset("bigbio/quaero", name="quaero_medline_bigbio_kb")

    preprocessor = NestedPerDepthNERPreprocessor(data, "fr")

    data_train = preprocessor.process_data("train")
    data_val = preprocessor.process_data("validation")
    data_test = preprocessor.process_data("test")

    data_module = NestedNERDataModule(data_train, data_val, batch_size=4, max_length=512)

    model = NestedPerDepthNERModel(
        n_depth=preprocessor.n_layers, id2label=preprocessor.id2label
    ) 

    trainer = pl.Trainer(
        max_epochs=50,
        logger=wandb_logger,
        #log_every_n_steps=20,
        precision="16",
        fast_dev_run=False,
    )

    trainer.fit(model, data_module)

if __name__ == "__main__":
    main()