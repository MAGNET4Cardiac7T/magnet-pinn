import torch
def worker_init_fn(worker_id):
     worker_info = torch.utils.data.get_worker_info()
     dataset = worker_info.dataset  # the dataset copy in this worker process
     overall_simulations = dataset.simulation_list
     num_workers = worker_info.num_workers

     # configure the dataset to only process the split workload
     worker_id = worker_info.id
     worker_simulations = overall_simulations[worker_id::num_workers]
     dataset.simulation_list = worker_simulations