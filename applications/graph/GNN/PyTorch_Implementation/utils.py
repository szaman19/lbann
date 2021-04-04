import os
import os.path as osp
import time
import statistics
import torch


def get_num_gpus_per_node():
  """Number of GPUs on this node"""
  return torch.cuda.device_count()

def get_local_rank():
  """Get local rank from environment"""

  # If run with SLURM
  if 'SLURM_LOCALID' in os.environ:
    return int(os.environ['SLURM_LOCALID'])

  return 0

def get_local_size():
  """Get local size from environment"""

  if 'SLURM_NTASKS_PER_NODE' in os.environ:
    return int(os.environ['SLURM_NTASKS_PER_NODE'])
  
  return 1

def get_world_rank():
  """Get global rank"""
  if 'SLURM_PROCID' in os.environ:
    return int(os.environ['SLURM_PROCID'])
  
  return 0

def get_world_size():
  """Get global work size"""

  if 'SLURM_NTASKS' in os.environ:
    return int(os.environ['SLURM_NTASKS'])
  
  return 1

def init_dist(init_file):
  """Initialize PyTorch distributed backed"""
  torch.cuda.init()

  torch.cuda.set_device(0)

  init_file = osp.abspath(init_file)

  torch.distributed.init_process_group(backend='nccl',
                     init_method=f'file://{init_file}',
                     rank=get_world_rank(),
                     world_size=get_world_size())
  torch.distributed.barrier()

  # Must delete init file 
  if get_world_rank() == 0 and osp.exists(init_file):
    os.unlink(init_file)


class AverageTracker:
    """Keeps track of the average of a value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear the tracker."""
        self.vals = []

    def update(self, val, n=1):
        """Add n copies of val to the tracker."""
        self.vals.extend([val]*n)

    def mean(self):
        """Return the mean."""
        if not self.vals:
            return float('nan')
        return statistics.mean(self.vals)
