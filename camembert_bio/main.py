import hydra
from omegaconf import DictConfig
import wandb
from datasets import load_dataset
from data_handling.ner_dataset import NERPreprocessor, NERDataModule
from data_handling.nested_ner_dataset import NestedPerDepthNERPreprocessor, NestedNERDataModule 
from models.nested_ner_bert import NestedPerDepthNERModel, NestedPerClassNERModel
from models.ner_bert import NERModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from camembert_bio.evaluation.evaluate_offsets import EvaluationCallback, evaluate_model
import logging

logging.getLogger('stanza').setLevel(logging.ERROR)

@hydra.main(config_path="../config", config_name="default", version_base="1.1")
def main(cfg: DictConfig):
    wandb_logger = WandbLogger(name=f"{cfg.model.version}_{cfg.model.pretrained_model_name}_{cfg.dataset.name}", project="camembert_bio")

    data = load_dataset("bigbio/quaero", cfg.dataset.name)

    preprocessor = NERPreprocessor(data, "fr")

    data_train = preprocessor.process_data("train")
    data_val = preprocessor.process_data("validation")
    data_test = preprocessor.process_data("test")

    data_module = NERDataModule(data_train, data_val, data_test, batch_size=8, max_length=512, num_workers=2)

    model_params = {
        "pretrained_model_name": cfg.model.pretrained_model_name,
        "learning_rate": cfg.model.learning_rate,
        "dropout_prob": cfg.model.dropout_prob,
        "id2label": preprocessor.id2label
    }

    if cfg.model.version == "PerDepth":
        model = NestedPerDepthNERModel(**model_params, n_depth=preprocessor.n_layers, stack_depths=True)
    elif cfg.model.version == "PerClass":
        model = NestedPerClassNERModel(**model_params, stack_classes=True)
    else:
        model = NERModel(**model_params)

    #evaluation_callback = EvaluationCallback(test_data=data["test"])

    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        #devices=find_usable_cuda_devices(1),
        precision=cfg.trainer.precision,
        fast_dev_run=cfg.trainer.fast_dev_run,
        #callbacks=[evaluation_callback],
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    #test = data["test"]
    #results = evaluate_model(model, test)
    #wandb_logger.log_metrics(results)

if __name__ == "__main__":
    main()
