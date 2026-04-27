import os
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import ot

from utils import *
from model import GraphUnetAttention

def train(args, model, train_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.end_learning_rate
    )

    loss_list = []
    runtime_list = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fnc = nn.MSELoss()

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch in train_dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)

            loss = loss_fnc(out, batch.y.float())
            print(loss)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_runtime = time.time() - start_time

        avg_loss = epoch_loss / len(train_dataloader)
        loss_list.append(avg_loss)
        runtime_list.append(epoch_runtime)

        scheduler.step()

        if (epoch) % args.save_freq == 0:
            model_save_path = os.path.join(args.exp_name, f"model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch} to {model_save_path}")
    
        print(f"Epoch {epoch}/{args.num_epochs} - Loss: {avg_loss:.7f}, Runtime: {epoch_runtime:.2f}s")

    results = {"loss": loss_list, "runtime": runtime_list}
    txt_path = f"{args.exp_name}/results.txt"
    pkl_path = f"{args.exp_name}/results.pkl"

    model_save_path = os.path.join(args.exp_name, f"model.pth")
    torch.save(model.state_dict(), model_save_path)

    with open(txt_path, "w") as f:
        for epoch, (loss, runtime) in enumerate(zip(loss_list, runtime_list)):
            f.write(f"Epoch {epoch+1}: Loss = {loss:.7f}, Runtime = {runtime:.2f}s\n")

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Training completed. Results saved to {txt_path} and {pkl_path}")


def sinkhorn_distance(pred, gt, reg=0.1, numItermax=500, stopThr=1e-5):
    n = pred.shape[0]
    a = b = np.ones(n) / n
    M_ab = ot.dist(pred, gt, metric='sqeuclidean')
    M_aa = ot.dist(pred, pred, metric='sqeuclidean')
    M_bb = ot.dist(gt,   gt,   metric='sqeuclidean')

    C_ab = ot.sinkhorn2(a, b, M_ab, reg=reg, numItermax=numItermax, stopThr=stopThr)
    C_aa = ot.sinkhorn2(a, a, M_aa, reg=reg, numItermax=numItermax, stopThr=stopThr)
    C_bb = ot.sinkhorn2(b, b, M_bb, reg=reg, numItermax=numItermax, stopThr=stopThr)

    D = C_ab - 0.5*(C_aa + C_bb)
    return max(D, 0)


def eval_pkl(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    all_loss = []
    all_ke = []
    all_sinkhorn = []

    dataset_path = os.path.join(args.dataset_root, "valid.pkl")

    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
        
    positions_list = data['positions']
    types_list = data['types']
    
    assert len(positions_list) == len(types_list), "Mismatch between number of position and type samples!"
    print(f"Successfully loaded {len(positions_list)} samples.\n")

    all_data = []
    classes = np.array([5, 6, 7])

    for i, (pos_sample, type_sample) in enumerate(zip(positions_list, types_list)):
        particle_information = (type_sample == classes).astype(int)
        pos = torch.from_numpy(pos_sample)
        particle_information = torch.from_numpy(particle_information)


        loss_fnc = nn.MSELoss()
        num_frames, num_particles, _ = pos.shape

        start_index = 0
        ls_loss = []
        ls_ke = []
        ls_sinkorn = []
        while start_index < num_frames-6:
            x = pos[start_index:start_index+6] # (6, 2500, 2)
            y = pos[start_index+6]  # Shape: (2500, 2)
                    
            velocity = x[1:, :, :] - x[:-1, :, :]
            velocity = vel_normalizer(velocity)
            velocity = velocity.permute(1, 0, 2).contiguous()
            velocity = velocity.view(num_particles, -1)

            boundary_features = get_boundary_features(x[-1, :, :])
                    
            node_feature = torch.cat([x[-1, :, :], velocity, boundary_features, particle_information], dim = -1)
            
            edge_index, edge_attr = create_edge(x[-1, :, :], args.connectivity_radius)
            data = Data(x=node_feature, pos=x[-1, :, :], edge_index=edge_index, edge_attr=edge_attr)
            data = data.to(device)
            
            pv = model(data)
            pv = pv.detach().cpu()
            pv = vel_unnormalizer(pv)
            output = x[-1, :, :] + pv

            bounds = [[0.1, 0.9], [0.1, 0.9]]
            bounds_tensor = torch.tensor(bounds, device=output.device) 
            min_bounds = bounds_tensor[:, 0] # Tensor: [0.1, 0.1]
            max_bounds = bounds_tensor[:, 1] # Tensor: [0.9, 0.9]
            output = torch.clamp(output, min=min_bounds, max=max_bounds)

            loss = loss_fnc(output, y)
            ls_loss.append(loss.item())

            pred_vel = output - x[-1, :, :]
            gt_vel = y - x[-1, :, :]

            pred_ke = 0.5 * torch.sum(pred_vel**2, dim=0)
            gt_ke = 0.5 * torch.sum(gt_vel**2, dim=0)
            loss_ke = loss_fnc(pred_ke, gt_ke)
            ls_ke.append(loss_ke.item())

            sinkhorn = sinkhorn_distance(output.numpy(), y.numpy())
            ls_sinkorn.append(sinkhorn)

            print(f"start_index: {start_index}, loss: {loss}, loss_ke: {loss_ke}, sinkhorn: {sinkhorn}")

            pos[start_index+6] = output
                    
            start_index += 1

        print(ls_loss)
        print(ls_ke)
        print(ls_sinkorn)

        all_loss.append(ls_loss)
        all_ke.append(ls_ke)
        all_sinkhorn.append(ls_sinkorn)

        break

    
    all_loss = np.array(all_loss)
    print(f"all_loss: {np.mean(all_loss, axis=0).tolist()}")
    np.savetxt(os.path.join(args.exp_name, "mse.txt"), all_loss, fmt='%.18e')
    
    all_ke = np.array(all_ke)
    print(f"all_ke: {np.mean(all_ke, axis=0).tolist()}")
    np.savetxt(os.path.join(args.exp_name, "ke.txt"), all_ke, fmt='%.18e')

    all_sinkhorn = np.array(all_sinkhorn)
    print(f"all_sinkhorn: {np.mean(all_sinkhorn, axis=0).tolist()}")
    np.savetxt(os.path.join(args.exp_name, "sinkhorn.txt"), all_sinkhorn, fmt='%.18e')




def main():
    parser = argparse.ArgumentParser(description="Argument parser for specifying dataset, model, and training configurations")

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--data_save_path", type=str)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--split_interval", type=int)
    parser.add_argument("--connectivity_radius", type=float)

    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--end_learning_rate", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--save_freq", type=int)
    parser.add_argument("--to_train", action="store_true")
    parser.add_argument("--model_path", type=str)

    parser.add_argument("--output_size", type=int)
    parser.add_argument("--latent_size", type=int)
    parser.add_argument("--node_input_size", type=int)
    parser.add_argument("--edge_input_size", type=int)
    parser.add_argument("--bottom_steps", type=int)
    parser.add_argument("--down_steps", type=int)
    parser.add_argument("--up_steps", type=int)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--l_n", type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(args.exp_name):
        os.makedirs(args.exp_name)
        print(f"Folder '{args.exp_name}' created.")
    else:
        print(f"Folder '{args.exp_name}' already exists.")

    model = GraphUnetAttention(args)

    print(f"args.to_train: {args.to_train}")
    if args.to_train:
        train_dataloader = load_train(args)
        train(args, model, train_dataloader)
    else:
        with torch.no_grad():
            # eval(args, model)
            eval_pkl(args, model)

if __name__ == "__main__":
    main()

