
from typing import Iterable
import torch
import time
from tqdm import tqdm

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                     epoch: int, args = None, logger = None, writer=None):
    model.train()
    criterion.train()

    pbar = tqdm(total=len(data_loader.dataloader), desc=f"Initial Loss Fused: Pending")
    for i, data in enumerate(data_loader):
        samples = data['image'].to(torch.device(args.device))
        targets = data['label'].to(torch.device(args.device))

        output = model(samples)
        logits = output["logits"] if isinstance(output, dict) else output
        try:
            loss_out = criterion(logits, targets.float(), epoch=epoch)
        except TypeError:
            loss_out = criterion(logits, targets.float())

        if isinstance(loss_out, dict):
            loss_final = loss_out["loss_total"]
            loss_boundary = loss_out["loss_boundary"].item()
        else:
            loss_final = loss_out
            loss_boundary = None
        cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))

        loss_final_str = '{:.4f}'.format(loss_final.item())
        l = optimizer.param_groups[0]['lr']
        if isinstance(loss_out, dict):
            logger.info(
                f"time -> {cur_time} | Epoch -> {epoch} | image_num -> {data['A_paths']} | "
                f"loss total -> {loss_final_str} | loss bce -> {loss_out['loss_bce'].item():.4f} | "
                f"loss dice -> {loss_out['loss_dice'].item():.4f} | "
                f"loss boundary -> {loss_boundary:.4f} | lr -> {l}"
            )
        else:
            logger.info(f"time -> {cur_time} | Epoch -> {epoch} | image_num -> {data['A_paths']} | loss final -> {loss_final_str} | lr -> {l}")

        if loss_boundary is None:
            pbar.set_description(f"Loss: {loss_final.item():.4f}")
        else:
            pbar.set_description(
                f"Loss: {loss_final.item():.4f} Boundary: {loss_boundary:.4f}"
            )
        pbar.update(1)
        optimizer.zero_grad()
        loss_final.backward()
        optimizer.step()

        # global_step = epoch * len(data_loader.dataloader) + i
        # writer.add_scalar("Loss/train", loss_final.item(), global_step)

    pbar.close()

