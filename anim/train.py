
#External
import argparse
import logging
import os
import numpy as np 
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch 
torch.autograd.set_detect_anomaly(True)

#Internal
from anim.data import amass 
import anim.models as models

__TRAIN_INFO_NAME = 'train_info.json'

def get_model_train_info(model_dir: str):
    os.path.isdir(model_dir)
    train_info_path = os.path.join(model_dir, __TRAIN_INFO_NAME)
    if not os.path.exists(train_info_path):
        logging.info(f'{train_info_path} does not exists, then ignored')
        return None
    else:
        with open(train_info_path, 'r') as fp:
            train_info = json.load(fp)
        return train_info

def __main():

    # Overwrite log file every time the script runs
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()

    # Global args
    parser.add_argument('model', type=str,
                        help="Either a path to a model config TOML to start training,"
                             "or a model directory to continue training from")
    parser.add_argument('--save_dir', type=str, default='./saves',
                        help="saving directory, only for models trained from scratch")
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--dataset', type=str,
                        choices=('amass-p1', 'amass-p2', 'egobody'), default='amass-p1')
    parser.add_argument('--win_len', type=int, default=40)
    parser.add_argument('--win_overlap', type=int, default=5)
    parser.add_argument('--zero_betas', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--data_ratio', type=float, default=None,
                        help="Ratio of recs to use for train/val datasets;"
                             "only approximate as doesn't account for clip lengths")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--epochs_per_save', type=int, default=10)
    parser.add_argument('--device_str', type=str, default='cuda')
    parser.add_argument('--dtype_str', type=str, default='float32')

    args = parser.parse_args()

    # --- Prepare for training ---
    device = torch.device(args.device_str)
    if args.dtype_str == 'float32':
        dtype = torch.float32
    else:
        raise NotImplementedError
    
    # --- Create/load model ---
    training_from_scratch = os.path.isfile(args.model)
    if training_from_scratch:
        model_cfg = args.model
        model, model_dir = models.create_new_model_and_dir(model_cfg, args.save_dir)
        last_epoch = -1
        logging.info(f"Training '{model_dir}' from scratch")
    else:
        assert os.path.isdir(args.model), "'model' was not an existing file or a directory"
        model_dir = args.model
        # TODO Fetch/save best epoch instead of fetching last?
        last_epoch = models.get_last_model_epoch(model_dir)
        model = models.load_model(model_dir, last_epoch)
        logging.info(f"Training '{model_dir}' from from epoch {last_epoch}; note that train_info.json will be overwritten")

    # Save training arguments
    train_info = {
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'win_len': args.win_len,
        'win_overlap': args.win_overlap,
        'zero_betas': args.zero_betas,
        'data_ratio': args.data_ratio
    }
    train_info_path = os.path.join(model_dir, __TRAIN_INFO_NAME)
    with open(train_info_path, 'w') as fp:
        json.dump(train_info, fp, indent=4)

    model = model.to(device, dtype)

    # --- Dataset preparation ---
    train_dataset = amass.get_dataset(args.dataset, 'train', args.data_ratio,
                                     win_len=args.win_len, win_overlap=args.win_overlap, zero_betas=args.zero_betas, dtype=dtype)
    val_dataset = amass.get_dataset(args.dataset, 'test', args.data_ratio,
                                   win_len=args.win_len, win_overlap=args.win_overlap, zero_betas=args.zero_betas, dtype=dtype)
    # Need drop last due to loss averaging (fixed batch size)
    train_dataloader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, args.batch_size,
                                shuffle=True, num_workers=args.dataloader_num_workers, drop_last=True)
    
    logging.info(f'Length (train | valid): {len(train_dataloader)} | {len(val_dataloader)}')

    log_dir: str = os.path.join(model_dir, 'logs')
    log_writer = SummaryWriter(log_dir)
    # --- Main Loop --- 
    error_last_save = False
    epoch = last_epoch + 1
    try:
        while epoch < args.max_epoch:

             # --- Train ---
            model.train()
            
            train_losses = None
            for train_batch in train_dataloader:
                with torch.no_grad():
                    model_input, model_target = models.batch_to_model_input_and_target(
                        train_batch, device, dtype, mode3d=model.mode3d)
                    
                model.reset()
                batch_train_losses = model.forward_pass(model_input, model_target, optimise=True)

                if train_losses is None:
                    train_losses = {loss_str: [] for loss_str in batch_train_losses}
                for loss_str in train_losses:
                    train_losses[loss_str].append(batch_train_losses[loss_str].detach().cpu().numpy())

            # --- Validation ---

            model.eval()

            val_losses = None
            for val_batch in val_dataloader:
                with torch.no_grad():
                    model_input, model_target = models.batch_to_model_input_and_target(
                        val_batch, device, dtype, mode3d=model.mode3d)

                    model.reset()
                    batch_val_losses = model.forward_pass(model_input, model_target)

                if val_losses is None:
                    val_losses = {loss_str: [] for loss_str in batch_val_losses}
                for loss_str in val_losses:
                    val_losses[loss_str].append(batch_val_losses[loss_str].detach().cpu().numpy())

            # --- Log ----

            for loss_str in train_losses:
                train_losses[loss_str] = np.mean(train_losses[loss_str])
                val_losses[loss_str] = np.mean(val_losses[loss_str])
                logging.info(f"Epoch={epoch} | {loss_str}: train={train_losses[loss_str]:.6f} | val={val_losses[loss_str]:.6f}")

                try:
                    log_writer.add_scalars(
                        loss_str, {'train': train_losses[loss_str], 'val': val_losses[loss_str]}, epoch)
                except OSError:
                    pass  # Ignore I/O error so skip this epoch log

            # --- Save model ---

            if error_last_save or (epoch % args.epochs_per_save == 0):
                error_last_save = False
                try:
                    models.save_model(model, model_dir, epoch)
                except OSError:
                    print(f"Experienced OSError attempting to save model during epoch {epoch}")
                    error_last_save = True  # Ignore I/O error but try to save next epoch

            model.epoch_end(epoch, train_losses, val_losses)
            epoch += 1

    except KeyboardInterrupt:
        print("Ending run, but saving model first")
    except models.StopTrainingException as e:
        print("Training interrupted:", e)
    models.save_model(model, model_dir, epoch)

    print(f"Finished training {model_dir} at epoch {epoch}")

    
if __name__ == '__main__':
    __main()