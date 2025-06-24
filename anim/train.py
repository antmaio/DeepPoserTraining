
#External
import argparse
import logging
import os

from torch.utils.data import DataLoader
import torch 

#Internal
from anim.data import amass 
import anim.models as models 

def __main():


    # Overwrite log file every time the script runs
    logging.basicConfig(
        filename='anim_process_log.txt',
        filemode='w',  # 'w' = overwrite
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
                        choices=('amass-p1', 'amass-p2', 'egobody'), default='amass-p2')
    parser.add_argument('--win_len', type=int, default=20)
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

    # --- Main Loop --- 
    epoch = last_epoch + 1
    try:
        while epoch < args.max_epoch:
            model.train()
            train_losses = None
            for train_batch in train_dataloader:
                with torch.no_grad():
                    model_input, model_target = models.batch_to_model_input_and_target(
                        train_batch, device, dtype)
                    logging.info('pass ok')
                    #_ = model(model_input)
                
    except KeyboardInterrupt:
        logging.info("Ending run, but saving model first")
    except models.StopTrainingException as e:
        logging.info("Training interrupted:", e)
    models.save_model(model, model_dir, epoch)
    
if __name__ == '__main__':
    __main()