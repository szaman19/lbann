
from utils import get_world_size, init_dist, AverageTracker
import torch
import argparse
import time

from torch_geometric.data import DataLoader


from ogb.utils import smiles2graph
from ogb.lsc import PygPCQM4MDataset
from LSC_Trainer import LSC_Trainer


desc = "PyTorch Geometric Distributed Trainer for OGB LSC dataset"

parser = argparse.ArgumentParser(description=desc)

parser.add_argument(
    '--mini-batch-size', action='store', default=2048, type=int,
    help='mini-batch size (default: 2048)', metavar='NUM')

parser.add_argument('--no-sync', dest='mini_batch_sync', action='store_false')

parser.add_argument('--sync', dest='mini_batch_sync', action='store_true')


parser.add_argument('--dist', dest='dist', action='store_true')

parser.set_defaults(feature=True)
args = parser.parse_args()

mb_size = args.mini_batch_size

sync = args.mini_batch_sync

distributed_training = args.dist

ROOT = '/p/vast1/zaman2/PyG_Data'  # Location of Dataset


def main(BATCH_SIZE, dist=False, sync=True):
    time_stamp = time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())

    if dist:
        init_dist("/p/vast1/zaman2/randevous_files_"+time_stamp)
        rank = torch.distributed.get_rank()

    else:
        rank = 0

    primary = rank == 0
    world_size = get_world_size()

    if primary:
        print("Running distributed: ", dist, "\t world size: ", world_size)

    train_dataset = PygPCQM4MDataset(root=ROOT, smiles2graph=smiles2graph)
    split_idx = train_dataset.get_idx_split()

    train_dataset = train_dataset[split_idx["train"]]

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(.9 * len(train_dataset)), len(train_dataset)-int(.9*len(train_dataset))])

    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=world_size,
                                                                        rank=rank)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=(BATCH_SIZE // world_size),
                              sampler=train_sampler,
                              pin_memory=True)

    if dist:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSC_Trainer().to(device)

    if dist:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[0],
                                                          output_device=0,
                                                          find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = torch.nn.MSELoss()

    if primary:
        file_name = "MB_"+str(BATCH_SIZE) + "_" + world_size +".log"

        logger = open(file_name, 'w')

        print("Writing log information to ", file_name, flush=True)

    for epoch in range(1, 3):

        epoch_loss = 0
        epoch_start_time = time.perf_counter()
        batch_times = AverageTracker()

        for i, data in enumerate(train_loader):

            _time_start = time.perf_counter()
            data = data.to(device)
            y = data.y

            pred = model(data)
            loss = criterion(y, pred.squeeze())
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()

            if rank == 0:
                batch_times.update(time.perf_counter() - _time_start)
                # print("Mini Batch Times ", i,": \t", batch_times.mean(), flush=True)

            if dist and sync:
                torch.distributed.barrier()  # This ensures that Global Mini Batches are synced

        if dist:
            torch.distributed.barrier()

        if primary:
            message = "Epoch {}: Total elapsed time {:.3f} \t Average Mini Batch Time {:.3f} \n"
            epoch_time = time.perf_counter() - epoch_start_time

            logger.write(message.format(epoch, epoch_time, batch_times.mean()))
            logger.flush()
            print(message.format(epoch, epoch_time, batch_times.mean()), flush=True)
    if dist:
        torch.distributed.barrier()

    if primary:
        logger.close()


if __name__ == '__main__':
    main(mb_size, distributed_training, sync)
