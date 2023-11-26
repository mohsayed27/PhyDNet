import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

root = 'D:\dev\SolarForescastingDataset'

class SolarForecasting(Dataset):
    def __init__(self, root, is_train= True, n_frames_input= 8, n_frames_output= 8, 
                transform=None, hdf5_name ='2017_2019_images_pv_processed.hdf5', 
                trainval_timestamps_name='times_trainval.npy', test_timestamps_name= 'times_test.npy', 
                dt= 2, tolerance= 0):
        '''
        dt: (number) delta time, difference between images (1/f)
        tolerance: (number) +/- time tolerance to the difference
        '''
        super(SolarForecasting, self).__init__()

        self.hdf5_path = os.path.join(root, hdf5_name)
        self.trainval_times_path = os.path.join(root, trainval_timestamps_name)
        self.test_times_path = os.path.join(root, test_timestamps_name)
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform

        # Open HDF5 file
        with h5py.File(self.hdf5_path, 'r') as file:
            if is_train:
                self.raw_data = file['trainval'] 
                self.timestamp = np.load(self.trainval_times_path, allow_pickle= True)
            else:
                self.raw_data = file['test']
                self.timestamp = np.load(self.test_times_path, allow_pickle= True)
            self.images = self.raw_data['images']
            self.pv = self.raw_data['pv_outputs']
            self.total_size = self.timestamp.shape[0]
            

        self.preprocessing(dt, tolerance)
        

        
    def preprocessing(self, dt= 2, tolerance= 0):
        """
        preprocessing process is to indicate the valid indices of the images going \
        over the timestamps of the data looping over the each image and for the number \
        of input and output frames for each sequence 
        """
        # Windows is the whole sequence length
        W = self.n_frames_total

        # empty list for valid sequences
        vld_sequence = []

        # index in dataset(i), timestamp(ts)
        for i, ts in enumerate(self.timestamp):
            # sequence starts with time stamp
            seq = [i]
            # apnd indicates sequence updated
            apnd = 1
            # vld indicates sequence has not been invalidated
            vld = 1
            # current timestamp
            curr_ts = ts
            # pointer to next timestamps
            j = i + 1
            # quit loop when sequence is complete
            while(len(seq) < self.n_frames_total):
                # quit condition when pointer is out of range
                if j >= self.total_size:
                    vld = 0
                    break
                # update prev ts on a sequence update
                if apnd == 1:
                    prev_ts = curr_ts
                # timestamp according to pointer
                curr_ts = self.timestamp[j]
                # clearing apnd to check it 
                apnd = 0
                # quit when timestamp is greater than time between frames
                # timestamps are non decreasing
                # tolerance = 0
                if (curr_ts - prev_ts).total_seconds() > 60*dt + tolerance:
                    vld = 0
                    break
                # append when timestamp equals time between frames
                # tolerance = 0 
                elif (curr_ts - prev_ts).total_seconds() >= 60*dt - tolerance:
                    seq.append(j)
                    apnd = 1
                # advance pointer 
                j += 1
            # if not invalidated, sequence is accepted
            if vld == 1: vld_sequence.append(seq)
        # saving valid sequences into the class
        self.sequences = vld_sequence
        self.length = len(vld_sequence)
            
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get indices of sequence
        indices = self.sequence[idx]
        # get input indices (beginning)
        input_idcs = indices[:self.n_frames_input]
        # get output indices (ending)
        output_idcs = indices[-self.n_frames_output:]

        # get input images
        input_frames = np.array([self.images[i] for i in input_idcs])
        # get input images
        input_pv = np.array([self.pv[i] for i in input_idcs])
        # get output pv
        output_frames = np.array([self.images[i] for i in output_idcs])
        # get output pv
        output_pv = np.array([self.pv[i] for i in output_idcs])

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # transform into torch tensors
        input_frames = torch.from_numpy(input_frames).contiguous()
        output_frames = torch.from_numpy(output_frames).contiguous()
        input_pv = torch.from_numpy(input_pv).contiguous()
        output_pv = torch.from_numpy(output_pv).contiguous()
        
        return [idx, input_frames, output_frames, input_pv, output_pv]

# Example usage:
trainval_dataset = SolarForecasting(root= root)
trainval_dataloader = DataLoader(trainval_dataset, batch_size=64, shuffle=True)
