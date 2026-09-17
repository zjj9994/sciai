"""End-to-end sciai reference API demonstration."""

from sciai import AutoDataset, AutoModel, ScienceTrainer, TrainingConfig, pipeline
from sciai.training import Adam

model = AutoModel.from_pretrained("sciai/molecule-baseline")
dataset = AutoDataset.load("sciai/toy-molecules")
trainer = ScienceTrainer(
    config=TrainingConfig(epochs=100, batch_size=len(dataset)),
    optimizer=Adam(learning_rate=0.01),
)
trainer.fit(model, dataset)

predictor = pipeline("property-prediction", model=model)
print(predictor("CCO"))
