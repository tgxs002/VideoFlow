import sys
sys.path.append('core')
import os
import argparse
import torch
import json
import math
from tqdm import tqdm

from core.Networks import build_network

from utils.utils import InputPadder
from configs.multiframes_sintel_submission import get_cfg
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def prepare_image(video_path, target_resolution=432):

    images = []

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame
        height, width, _ = frame.shape
        if height < width:
            new_height = target_resolution
            new_width = int(width * (new_height / height))
        else:
            new_width = target_resolution
            new_height = int(height * (new_width / width))
        frame = cv2.resize(frame, (new_width, new_height))

        frame = torch.from_numpy(frame).permute(2, 0, 1).float()
        images.append(frame)

    cap.release()

    return torch.stack(images)

class video_dataset(torch.utils.data.Dataset):
    def __init__(self, instruction_file, input_folder, output_folder, target_resolution=432):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_resolution = target_resolution

        self.data = []
        # instruction file is a json
        with open(instruction_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        input_video = os.path.join(self.input_folder, data_item['input_video'])
        output_file = os.path.join(self.output_folder, data_item['output_file'])

        return dict(
            frames=prepare_image(input_video, self.target_resolution),
            output_file=output_file
        )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', default='default')
    parser.add_argument('--instruction_file', required=True, help='path to the instruction file (a json contain the input video path and output file path).')
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--target_resolution', type=int, default=128)
    parser.add_argument('--batch_size', default=30, type=int, help='number of frames to process in a batch.')
    parser.add_argument('--padding_frames', default=4, type=int, help='number of frames to pad at the beginning and end of the video.')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers to use for data loading.')
    
    args = parser.parse_args()
    assert args.padding_frames % 2 == 0, "padding_frames must be even."

    cfg = get_cfg()
    cfg.update(vars(args))

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.to(torch.device(local_rank))
    model.eval()

    if rank == 0:
        print(cfg.model)
        print("Parameter Count: %d" % count_parameters(model))

    # build dataloader
    dataset = video_dataset(cfg.instruction_file, cfg.input_folder, cfg.output_folder, cfg.target_resolution)
    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, sampler=sampler, collate_fn=lambda x: x)
    
    with torch.no_grad():

        for data in tqdm(dataloader):
            input_images = data[0]['frames']
            output_file = data[0]['output_file']
            input_images = input_images[None].to(torch.device(local_rank))
            padder = InputPadder(input_images.shape)
            input_images = padder.pad(input_images)
            # repeat the first frame and the last frame, so the model can predict both forward and backward flow of the first and last frame
            input_images = torch.cat([input_images[:, 0:1], input_images, input_images[:, -1:]], dim=1)
            
            if cfg.batch_size < input_images.size(1):
                num_batches = int(math.ceil((input_images.size(1) - cfg.padding_frames) / (cfg.batch_size - cfg.padding_frames)))
                flow_pre = []
                for batch_idx in range(num_batches):
                    print(f"Processing batch {batch_idx} / {num_batches}...")
                    batch_images = input_images[:, batch_idx * (cfg.batch_size - cfg.padding_frames) : (batch_idx + 1) * (cfg.batch_size - cfg.padding_frames) + cfg.padding_frames]
                    batch_flow, _ = model.module(batch_images, {})
                    batch_flow = padder.unpad(batch_flow[0]).cpu()
                    batch_flow = batch_flow.unflatten(0, (2, -1))
                    if len(flow_pre) > 0:
                        flow_pre[-1] = flow_pre[-1][:, :- (cfg.padding_frames - 2) // 2]
                        batch_flow = batch_flow[:, (cfg.padding_frames - 2) // 2:]
                    flow_pre.append(batch_flow)
                flow_pre = torch.cat(flow_pre, dim=1)
            else:
                # patch 
                continue
                batch_flow, _ = model.module(input_images, {})
                batch_flow = padder.unpad(batch_flow[0]).cpu()
                flow_pre = batch_flow.unflatten(0, (2, -1))

            # remove the redundant prediction of the first and last frame
            forward_flow = flow_pre[0, :-1]
            backward_flow = flow_pre[1, 1:]
            flow_pre = torch.stack([forward_flow, backward_flow], dim=0)

            # the first is forward flow, the second is backward flow
            # convert from torch tensor to numpy array
            flow_pre = flow_pre.cpu().numpy().astype(np.float16)

            # save as npy
            np.save(output_file, flow_pre)