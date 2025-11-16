from itertools import islice
import logging
from pathlib import Path
import random
import shutil
import tempfile

import hydra
from omegaconf import DictConfig, OmegaConf

import padertorch as pt
from padertorch.train.hooks import LRAnnealingHook
import torch
from tqdm.auto import tqdm

from .modules.data_loader import JsonParser, Dataloader

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="default")  #hydra sucht/lädt nach conf/default.yaml
def main(config: DictConfig):   #config als OmegaConf DictConfig
    OmegaConf.resolve(config)   #resolve macht aus variablen zahlen wie z.b. decay_factor: ${min_lr}/${lr} -> 0.01
    _config = OmegaConf.to_container(config, resolve=True) #macht ne dict aus Hydra-OmegaConf
    _config['trainer'] = pt.Trainer.get_config(_config['trainer']) #sorgt für gültige config

    if config.trainer.storage_dir is None: #speicherort überprüfen / neuen erzeugen:
        # Create new storage directory
        if config.launch.dry_run: #wenn dry_run = true in default.yaml 
            storage_dir = Path(tempfile.mkdtemp()) #keine logs, checkpoints (testen)
        else:
            storage_dir = pt.io.get_new_subdir( #pt erzeugt neue checkpoints ordner
                basedir=Path(config.base_dir),  # in base_dir (default.yaml)
            )
            _config["trainer"]["storage_dir"] = str(storage_dir) #_config - python-dict
        config.trainer.storage_dir = str(storage_dir)            #config - Hydra
    config_file = Path(config.trainer.storage_dir) / 'config.yaml'
    if config.launch.train and not config_file.exists():
        pt.io.dump_config(_config, config_file)     #wenn launch.train = true speicher gesamte config in /storage_dir/config.yaml

    if config.launch.resume: #resume aus default.yaml | false: neues training / true: bestehendes fortsetzn
        if config.launch.dry_run: # nicht bei dry run 
            raise RuntimeError("Cannot resume in dry run mode.")
        config = OmegaConf.load(
            Path(config.trainer.storage_dir) / 'config.yaml' #config.yaml wird geladen statt default.yaml (wegen resume)
        )

    log.info(OmegaConf.to_yaml(config)) #parameter info in konsole anzeigen (um zu sehen obs klappt)

    parsers = []
    for _, db_conf in config.databases.items(): #schaut bei default.yaml nach databases: bvcc/nisqa (factory, json usw.)
        db_conf = OmegaConf.to_container(db_conf, resolve=True) #OmegaConf->dict (JsonParser.from_config erwartet dict)
        parsers.append(JsonParser.from_config(db_conf)) #erzeugt parsers für data_loader

    trainer_config = OmegaConf.to_container(config.trainer, resolve=True) #dict machen für padertorch
    trainer_config = pt.Trainer.get_config(trainer_config) # macht config padertorch kompatibel
    trainer = pt.Trainer.from_config(trainer_config) #trainer wird hier gebaut! (modell, optimizer, hooks ...)

    train_dataloader = Dataloader.from_config(
        OmegaConf.to_container(config.train_dataloader, resolve=True) #dict machen + train_dataloader aus default.yaml
    )(
        *parsers, # '*' entpackt die liste in funktionsargumente (mehrere datasets kombinieren)
        prepare_example_fn=getattr(trainer.model, "prepare_example", None), #prepare_example() nutzen wenn im modell vorhanden
    )                                                                       #vorhanden im ssl_mos!

    #-||- nur für val statt train
    val_dataloader = Dataloader.from_config(
        OmegaConf.to_container(config.val_dataloader, resolve=True)
    )(
        *parsers,
        prepare_example_fn=getattr(trainer.model, "prepare_example", None),
    )

#device auswählen
    if config.launch.accelerator == "auto": #auto in default.yaml 
        device = 0 if torch.cuda.is_available() else "cpu" #GPU bevorzugt
    else:
        device = config.launch.accelerator #zum manuel auswählen

    decay_factor = config.min_lr/config.lr #decay factor von default.yaml übernehmen (lr: 1e-4 / min_lr: 1e-6 = 0.01)
    trainer.register_hook(
        LRAnnealingHook(
            [1, config.trainer.stop_trigger[1]], #default.yaml: stop_trigger: [100, 'epoch'] also [1, 'epoch] -> LearningRate-Scheduler jede epoche ausführen
            [(config.trainer.stop_trigger[0], decay_factor)], #stop_trigger[0]=100, decay_factor=0.01 also [(100, 0.01)]: am ende 100er epoche soll lr=0.01 sein
            config.trainer.stop_trigger[1], #'epoch' 
        ),
    )

#test run vor BVCC/NISQA
    if config.launch.test_run: #wenn default.yaml 'test_run: true' (grad nich)
        trainer.test_run(train_dataloader, val_dataloader, device=device) #mini 'trainingslauf'
        # Test if model can be loaded
        pt.io.dump_config(
            _config, Path(trainer.storage_dir) / 'config.yaml'
        )
        (trainer.storage_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint()
        pt.Model.from_storage_dir( #checkpoint laden
            trainer.storage_dir, config_name="config.yaml",
            checkpoint_name="ckpt_latest.pth"
        )


#das eig training
    if config.launch.train: #default.yaml: 'train: true'
        if (
            not config.launch.resume
            and hasattr(trainer.model, "test_seed") #methode in ssl_mos: 'test_seeds(...)'
            and config.num_seeds > 0 #default.yaml: num_seeds: 100
        ):
            # Test seeds
            log.info("Testing seeds.")
            best_metric = None
            best_seed = None
            seed = torch.initial_seed() #aktuelle seed holen
            if isinstance(device, list):
                _device = device[0]
            else:
                _device = device
            trainer.model.to(_device).eval() #nur validierung nutzen
            subiterator = list(islice(
                iter(val_dataloader),
                int(
                    config.sub_iterator_length
                    / config.val_dataloader.batch_size
                )
            ))
            for _ in tqdm(range(config.num_seeds), desc="Testing seeds"):
                summary = trainer.model.test_seed(subiterator, seed, device)
                metric = summary["scalars"][config.validation.metric]
                if (
                    best_metric is None
                    or (config.validation.maximize and metric > best_metric)
                    or (not config.validation.maximize and metric < best_metric)
                ):
                    best_metric = metric
                    best_seed = seed
                    log.debug(
                        "New best %s: %f", config.validation.metric, best_metric
                    )
                del summary
                seed = torch.seed()

            del subiterator
            if best_seed is not None:
                torch.manual_seed(best_seed)
                torch.cuda.manual_seed(best_seed)
                random.seed(best_seed)
                trainer.model.reset_parameters(best_seed) #modell neu aufgabeun mir optimalen seed
                log.info(
                    "Finished testing seeds. Best %s: %f",
                    config.validation.metric, best_metric
                )
        if not config.launch.resume:
            try:
                if trainer.model.bilstm.zero_init_():
                    log.info(
                        "zero init: setting Transformer MLP weights to zeros."
                    )
            except AttributeError:
                pass
            try:
                if trainer.model.comparator.zero_init_():
                    log.info(
                        "zero init: setting Transformer MLP weights to zeros."
                    )
            except AttributeError:
                pass
            if config.zero_init:
                trainer.model.zero_init_()
                log.info(
                    "zero init: setting output layer weights to zeros."
                )

        #val laufen lassen, SRCC berechnen, LR reduzieren, checkpoint speichern, training stoppen
        if val_dataloader is not None:
            log.info("Registering validation hook.")
            trainer.register_validation_hook(
                val_dataloader, metric=config.validation.metric, 
                lr_update_factor=config.validation.lr_update_factor,
                n_back_off=config.validation.n_back_off,
                back_off_patience=config.validation.back_off_patience,
                maximize=config.validation.maximize,
            )
        trainer.train(                                          #TRAINING STARTEN | für jede epoche: batches laden, modell forwaard, 
            train_dataloader, resume=config.launch.resume,                       #| loss berechnen, backprop, optimizer step, 
            device=device,                                                       #| hooks ausführen, validation durchführen
        )                                                                           

    if config.launch.dry_run: #wenn default.yaml: 'dry_run: true'
        log.info("Dry run: Removing storage directory %s.", trainer.storage_dir)
        shutil.rmtree(config.trainer.storage_dir)


if __name__ == "__main__":
    main()
