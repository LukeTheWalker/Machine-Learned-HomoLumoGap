import os
import schnetpack as spk
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl

torch.set_float32_matmul_precision('high')

dataset = 'hl_gap_dataset'
if not os.path.exists(dataset):
    os.makedirs(dataset)

split_file = os.path.join(dataset, "split.npz")

if os.path.exists(split_file):
    os.remove(split_file)

cutoff = 100.

custom_data = spk.data.AtomsDataModule(
    './new_dataset.db',
    batch_size=10,
    distance_unit='Ang',
    # property_units={'homo':'eV', 'lumo':'eV'},
    property_units={'hl_gap':'eV'},
    num_train=1200,
    num_val=200,
    transforms=[
        trn.ASENeighborList(cutoff=cutoff),
        # trn.RemoveOffsets("homo", remove_mean=True),
        # trn.RemoveOffsets("lumo", remove_mean=True),
        trn.RemoveOffsets("hl_gap", remove_mean=True),
        trn.CastTo32()
    ],
    num_workers=11,
    pin_memory=True,
    split_file=split_file,
    # load_properties=["homo", "lumo"]
    # load_properties=["homo"]
    load_properties=["hl_gap"]
)

# means, stddevs = custom_data.get_stats("homo", divide_by_atoms=True, remove_atomref=False)
# print('Mean homo / atom:', means.item())
# print('Std. dev. homo / atom:', stddevs.item())

# means, stddevs = custom_data.get_stats("lumo", divide_by_atoms=True, remove_atomref=False)
# print('Mean lumo / atom:', means.item())
# print('Std. dev. lumo / atom:', stddevs.item())

custom_data.prepare_data()
custom_data.setup()

n_atom_basis = 300

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=1000, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=5,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

pred_homo = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="homo")
pred_lumo = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="lumo")
pred_hl   = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="hl_gap")

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    # output_modules=[pred_homo, pred_lumo],
    # output_modules=[pred_homo],
    output_modules=[pred_hl],
    # postprocessors=[trn.CastTo64(), trn.AddOffsets("homo", add_mean=True), trn.CastTo64(), trn.AddOffsets("lumo", add_mean=True)]
    # postprocessors=[trn.CastTo64(), trn.AddOffsets("homo", add_mean=True)]
    postprocessors=[trn.CastTo64(), trn.AddOffsets("hl_gap", add_mean=True)]
)

output_homo = spk.task.ModelOutput(
    name="homo",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

output_lumo = spk.task.ModelOutput(
    name="lumo",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

output_hl = spk.task.ModelOutput(
    name="hl_gap",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    # outputs=[output_homo, output_lumo],
    # outputs=[output_homo],
    outputs=[output_hl],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)

logger = pl.loggers.TensorBoardLogger(save_dir=dataset)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(dataset, "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=dataset,
    max_epochs=1000, # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=custom_data)