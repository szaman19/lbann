from utils import get_world_size, init_dist, AverageTracker, get_local_rank
import torch
import argparse
import time
import pickle
from torch_geometric.data import DataLoader


from SYNTH_Trainer import LSC_Trainer

import glob
desc = "PyTorch Geometric Distributed Trainer for OGB LSC dataset"

parser = argparse.ArgumentParser(description=desc)

parser.add_argument(
    '--mini-batch-size', action='store', default=2048, type=int,
    help='mini-batch size (default: 512)', metavar='NUM')

parser.add_argument(
    '--num-nodes', action='store', default=16, type=int,
    help='default 16', metavar='NUM')

parser.add_argument(
    '--num-edges', action='store', default=16, type=int,
    help='default 16', metavar='NUM')


parser.add_argument('--no-sync', dest='mini_batch_sync', action='store_false')

parser.add_argument('--sync', dest='mini_batch_sync', action='store_true')


parser.add_argument('--dist', dest='dist', action='store_true')

parser.set_defaults(feature=True)
args = parser.parse_args()

mb_size = args.mini_batch_size

num_nodes = args.num_nodes
num_edges = args.num_edges
sync = args.mini_batch_sync

distributed_training = args.dist



def main(BATCH_SIZE, dist=False, sync=True):
    time_stamp = time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())

    if dist:
        init_dist("/p/vast1/zaman2/randevous_files_"+str(BATCH_SIZE))
        rank = torch.distributed.get_rank()

    else:
        rank = 0

    primary = rank == 0
    world_size = get_world_size()

    if primary:
        print("Running distributed: ", dist, "\t world size: ", world_size)
    
    _files = [f"/p/vast1/zaman2/synth_data/{num_nodes}_{num_edges}_Pytorch.pickle"]
    #_files = glob.glob(_files_str)
    for _file in _files:
        
        #num_edges = _file.split("/")[-1].split(".")[0].split("_")[1]
        print(num_edges)
        with open(_file,'rb') as f:
            train_dataset = pickle.load(f)



        train_loader = DataLoader(train_dataset,
                                  batch_size=(BATCH_SIZE),
                                  pin_memory=True) 

        if dist:
            device = torch.device(f'cuda:{get_local_rank()}' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = LSC_Trainer(num_nodes).to(device)

        if dist:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              device_ids=[get_local_rank()],
                                                              output_device=get_local_rank())

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        criterion = torch.nn.MSELoss()

        if primary:
            file_name = "SYNTHETIC_LOGS/SYNTHETIC_"+str(num_nodes) + "_" + str(num_edges) +".log"

            logger = open(file_name, 'w')

            print("Writing log information to ", file_name, flush=True)

        for epoch in range(0, 5):

            epoch_loss = 0
            epoch_start_time = time.perf_counter()
            batch_times = AverageTracker()
            
            loss_tracker = AverageTracker()
           
            if (dist):
                train_loader.sampler.set_epoch(epoch)
            for i, data in enumerate(train_loader):

                _time_start = time.perf_counter()
                data = data.to(device)
                y = data.y

                pred = model(data)
                loss = criterion(y, pred)
                loss_tracker.update(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                #if dist and sync:
                    #torch.distributed.barrier()  # This ensures that Global Mini Batches are synced
                
                if rank == 0:
                    batch_times.update(time.perf_counter() - _time_start)
                    #print("Mini Batch Times ", i,": \t", batch_times.mean(), "LOSS: \t", loss_tracker.mean(), flush=True)
            if dist:
                torch.distributed.barrier()

                if primary:
                    message = "Epoch {}: Total elapsed time {:.3f} \t Average Mini Batch Time {:.3f} \n"
                    epoch_time = time.perf_counter() - epoch_start_time

                    logger.write(message.format(epoch, epoch_time, batch_times.mean()))
                    logger.flush()
                    print(message.format(epoch, epoch_time, batch_times.mean()), flush=True)
          
            else:
                if primary:
                    message = "Epoch {}: Total elapsed time {:.3f} \t Average Mini Batch Time {:.3f} \n"
                    epoch_time = time.perf_counter() - epoch_start_time

                    logger.write(message.format(epoch, epoch_time, batch_times.mean()))
                    logger.flush()
                    print(message.format(epoch, epoch_time, batch_times.mean()), flush=True)
            
        if primary:
            logger.close()


if __name__ == '__main__':
    main(mb_size, False, False)
