import numpy as np

from sciai import AutoDataset, AutoModel, ScienceTrainer, TrainingConfig
from sciai.domains.molecule import MolecularPropertyBaseline, Molecule
from sciai.training import Adam


def test_numpy_trainer_reduces_loss() -> None:
    dataset = AutoDataset.load("sciai/toy-molecules")
    model = MolecularPropertyBaseline()
    initial_loss, _ = model.loss_and_gradients(list(dataset))
    trainer = ScienceTrainer(
        config=TrainingConfig(epochs=200, batch_size=len(dataset), shuffle=False),
        optimizer=Adam(learning_rate=0.01),
    )
    history = trainer.fit(model, dataset)
    final_loss, _ = model.loss_and_gradients(list(dataset))
    assert history
    assert final_loss < initial_loss * 0.2


def test_safe_model_round_trip(tmp_path) -> None:
    model = MolecularPropertyBaseline(weights=[1, 2, 3, 4, 5], bias=0.5)
    expected = model.predict(Molecule.from_smiles("CCO"))
    artifact = model.save(tmp_path / "model")
    restored = AutoModel.from_pretrained(artifact)
    assert restored.predict("CCO") == expected
    assert (artifact / "model.json").is_file()
    assert (artifact / "weights.npz").is_file()


def test_auto_model_identifier() -> None:
    model = AutoModel.from_pretrained("sciai/molecule-baseline")
    assert isinstance(model, MolecularPropertyBaseline)
    assert np.isfinite(model.predict("CCO"))
