import os
import h5py
import pickle
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

noise_std = 6.7e-4


def vel_unnormalizer(normalized_vel):
    '''
    Input: (T, num_particles, 2)
    Output: (T, num_particles, 2)
    '''
    vel_mean = torch.tensor([6.839056577087033e-06, -0.0008123306585070839], 
                            device=normalized_vel.device, dtype=normalized_vel.dtype)
    vel_std = torch.tensor([0.0017002050274650662, 0.002129137009516546], 
                           device=normalized_vel.device, dtype=normalized_vel.dtype)

    # Inverse operation: multiply by std, then add mean
    vel = (normalized_vel * vel_std) + vel_mean
    
    return vel


def vel_normalizer(vel):
    '''
    Input: (T, num_particles, 2)
    Output: (T, num_particles, 2)
    '''
    vel_mean = torch.tensor([6.839056577087033e-06, -0.0008123306585070839], 
                            device=vel.device, dtype=vel.dtype)
    vel_std = torch.tensor([0.0017002050274650662, 0.002129137009516546], 
                           device=vel.device, dtype=vel.dtype)

    normalized_vel = (vel - vel_mean) / vel_std
    
    return normalized_vel

def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]

def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)

    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise

def create_edge(second_last_frame, connectivity_radius):
    diffs = second_last_frame.unsqueeze(1) - second_last_frame.unsqueeze(0)  # Shape: (2500, 2500, 2)
    distances = torch.norm(diffs, dim=-1)  # Shape: (2500, 2500)
        
    mask = (distances < connectivity_radius) & ~torch.eye(distances.size(0), dtype=torch.bool)  # Shape: (2500, 2500)
    edge_index = mask.nonzero(as_tuple=False).T
        
    src, dst = edge_index
    relative_positions = diffs[src, dst]  # Shape: (num_edges, 2)
    edge_distances = distances[src, dst].unsqueeze(1)  # Shape: (num_edges, 1)
    edge_attr = torch.cat((relative_positions, edge_distances), dim=-1)  # Shape: (num_edges, 3)

    return edge_index, edge_attr

def get_boundary_features(pos, bounds=[[0.1, 0.9], [0.1, 0.9]]):
    dist_to_walls = []
    for dim in range(pos.shape[-1]):
        dist_min = pos[:, dim] - bounds[dim][0] 
        dist_max = bounds[dim][1] - pos[:, dim] 
        dist_to_walls.extend([dist_min, dist_max])
    
    dist_to_walls = torch.stack(dist_to_walls, dim=-1)  # Shape: (num_particles, 4)
    
    return dist_to_walls

def create_subsequences(position, particle_information, sloshing_motion, seq_length, connectivity_radius, split_interval):
    num_frames, num_particles, _ = position.shape
    ls_data = []

    import random
    start_idx = random.randint(0, 9)
    while (start_idx + seq_length) < num_frames:
        subseq = position[start_idx : start_idx + seq_length]  # Shape: (7, 2500, 2)
        
        x = subseq[:-1] # (6, 2500, 2)
        y = subseq[-1] - subseq[-2]
        y = vel_normalizer(y)  # Shape: (2500, 2)

        sampled_noise = get_random_walk_noise_for_position_sequence(x.permute(1, 0, 2), noise_std_last_step=noise_std)
        sampled_noise = sampled_noise.permute(1, 0, 2)
        noised_x = x + sampled_noise
        
        velocity = noised_x[1:, :, :] - noised_x[:-1, :, :]
        velocity = vel_normalizer(velocity)
        velocity = velocity.permute(1, 0, 2).contiguous()
        velocity = velocity.view(num_particles, -1)

        boundary_features = get_boundary_features(noised_x[-1, :, :])
        
        node_feature = torch.cat([noised_x[-1, :, :], velocity, boundary_features, particle_information], dim = -1)

        edge_index, edge_attr = create_edge(x[-1, :, :], connectivity_radius)

        data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attr, y=y)

        ls_data.append(data)

        start_idx += split_interval

    return ls_data


def load_single_dataset(dataset_root, split, batch_size, seq_length, connectivity_radius, data_save_path, split_interval):
    dataset_path = os.path.join(dataset_root, "valid.pkl")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
        
    positions_list = data['positions']
    types_list = data['types']
    
    assert len(positions_list) == len(types_list), "Mismatch between number of position and type samples!"
    print(f"Successfully loaded {len(positions_list)} samples.\n")

    all_data = []
    classes = np.array([5, 6, 7])

    for i, (pos_sample, type_sample) in enumerate(zip(positions_list, types_list)):
        print(np.unique(np.array(type_sample)))
        particle_information = (type_sample == classes).astype(int)
        ls_data = create_subsequences(torch.from_numpy(pos_sample), torch.from_numpy(particle_information), None, seq_length, connectivity_radius, split_interval)
        all_data = all_data + ls_data
        print(f"len(all_data): {len(all_data)}")

    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    dataloader = DataLoader(all_data, batch_size=batch_size, shuffle=True)

    return dataloader


def load_train(args):
    dataset_root = args.dataset_root
    batch_size = args.batch_size
    seq_length = args.seq_length
    split_interval = args.split_interval
    connectivity_radius = args.connectivity_radius
    data_save_path = args.data_save_path
    split = "train.h5"

    file_path = f"{data_save_path}/{split}.pth"
    if True and os.path.exists(file_path):
        print(f"{split}.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = load_single_dataset(dataset_root, split, batch_size, seq_length, connectivity_radius, data_save_path, split_interval)
    else:
        print(f"{split}.pth does not exists. Create dataset")
        dataloader = load_single_dataset(dataset_root, split, batch_size, seq_length, connectivity_radius, data_save_path, split_interval)



    return dataloader