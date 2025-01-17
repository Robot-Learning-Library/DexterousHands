import wandb

def init_wandb(args):
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=args.wandb_name,
        monitor_gym=True,
        save_code=True,
    )