from preamble import *
from argument_parser import args

# config = load_config(os.path.join("configs", args.config + ".json"))
config = load_config(args.config)

TorchModel = import_from_module(config["model"]["module"], config["model"]["class"])
LitDataModule = import_from_module(config["datamodule"]["module"], config["datamodule"]["class"])
LitModel = import_from_module(config["litmodule"]["module"], config["litmodule"]["class"])


def main(save_name: str):
    ds = config["dataset"]
    ckpt_path = Path(f"saved_models/{ds}/{save_name}.ckpt")

    # Check if checkpoint exists and the overwrite flag is not set
    if ckpt_path.exists() and not args.overwrite:
        ckpt = str(ckpt_path)
    else:
        ckpt = None

    # Setup callbacks list for training
    callback_list = []
    if args.store_data:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_path.parent),  # Using parent directory of the checkpoint
            filename=save_name,
            monitor="val_loss",
            mode="min"
        )
        callback_list.append(checkpoint_callback)

    model = TorchModel(config["model"])
    datamodule = LitDataModule(args, config["datamodule"])
    lit_model = LitModel(model, config["training"])

    try:
        if torch.cuda.is_available() and args.use_cuda:
            devices = -1 if torch.cuda.device_count() > 1 else 1
            strategy = 'ddp' if devices == -1 else 'auto'
            accelerator = "auto"
        else:
            devices, strategy, accelerator = 1, 'auto', "cpu"

        if args.dry_run or not args.use_logger:
            logger = False
        else:
            run_name = f"{save_name}_{time.strftime('%d-%m_%H:%M:%S')}"
            task = config["task"]
            logger = WandbLogger(project=f"neural-stability-{task}", name=run_name)

        clip_val = config["training"]["clip"] if config["training"]["clip"] else None

        # Trainer configuration
        trainer = Trainer(max_epochs=config["training"]["epochs"],
                          logger=logger,
                          devices=devices,
                          strategy=strategy,
                          accelerator=accelerator,
                          callbacks=callback_list,
                          gradient_clip_val=clip_val,
                          fast_dev_run=args.dry_run,
                          enable_checkpointing=args.store_data)

        # Model fitting
        trainer.fit(lit_model, datamodule=datamodule, ckpt_path=ckpt)

    except Exception as e:
        print(f"An error occurred during setup: {e}")


if __name__ == "__main__":
    seed_everything(args.main_seed, workers=True)

    if args.scnd_seed is not None:
        config["datamodule"]["data_seed"] = args.scnd_seed

    if args.dry_run:
        config["datamodule"]["batch_size"] = 2

    config["model"]["stability_init"] = args.stability_init

    if args.add_name:
        add_name = f"_{args.add_name}"
    else:
        add_name = ""

    stability = "SII" if config["model"]["stability_init"] else "He"

    model_name = f'{config["task"]}_{config["dataset"]}_{stability}{add_name}'

    print(f'Preparing to train model: {model_name}')

    main(model_name)
