import os
import logging

import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import pandas as pd

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

logger = logging.getLogger(__name__)

class BaseEHRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        split,
        args,
        accelerator,
    ):
        self.args = args
        self.split = split
        self.structure = 'hi' # hierarchical structure
        self.seed = self.args.seed 
        self.data = data
        self.pred_tasks = self.args.pred_tasks
        self.data_prefix = data.split("_")[0]
        self.stay_id = {
            "mimiciii": "ICUSTAY_ID",
            "eicu": "patientunitstayid",
            "mimiciv": "stay_id"
        }[self.data_prefix]

        self.data_dir = os.path.join(self.args.input_path, self.data_prefix) 
        # Get split icustay ids
        self.fold_file = pd.read_csv(os.path.join(self.args.input_path, f"{data}_cohort.csv"))
        if self.args.extract_latent:
            self.hit_idcs = self.fold_file[self.fold_file[f'precision_{self.seed}'] == self.split][self.stay_id].values
        else:
            self.hit_idcs = self.fold_file[self.fold_file[f'split_{self.seed}'] == self.split][self.stay_id].values
        accelerator.print('loaded {} {} {} samples'.format(len(self.hit_idcs), data, self.split))

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class HierarchicalEHRDataset(BaseEHRDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_seq_len = self.args.max_seq_len
        self.max_word_len = self.args.max_word_len

    def __len__(self):
        return len(self.hit_idcs)

    def collator(self, samples):
        samples = [s for s in samples if s['input_ids'] is not None]
        if len(samples) == 0:
            return {}
        
        input = dict()
        out = dict()

        input['input_ids'] = [s['input_ids'] for s in samples]
        input['type_ids'] = [s['type_ids'] for s in samples]
        input['dpe_ids'] = [s['dpe_ids'] for s in samples]

        seq_sizes = []
        word_sizes = []
        for s in input['input_ids']:
            seq_sizes.append(s.shape[0])
            word_sizes.append(s.shape[1])

        target_seq_size = min(max(seq_sizes), self.max_seq_len) 
        target_word_size = min(max(word_sizes), self.max_word_len)

        collated_input = dict()
        for k in input.keys():
            collated_input[k] = torch.zeros(
                (len(input['input_ids']), target_seq_size, target_word_size,)
            ).long()

        for i, (seq_size, word_size) in enumerate(zip(seq_sizes, word_sizes)):
            seq_diff = seq_size - target_seq_size
            word_diff = word_size - target_word_size
            for k in input.keys():
                if k == 'input_ids':
                    prefix = 101
                elif k == "type_ids":
                    prefix = 5 
                elif k == "dpe_ids":
                    prefix = 0
                if word_diff < 0:
                    input[k][i] = np.pad(input[k][i], ((0,0), (0,-word_diff)), 'constant')

                if seq_diff == 0:
                    collated_input[k][i] = torch.from_numpy(input[k][i])
                elif seq_diff < 0:
                    padding = np.zeros((-seq_diff, target_word_size - 1,))
                    padding = np.concatenate(
                        [np.full((-seq_diff, 1), fill_value=prefix), padding], axis=1
                    )
                    collated_input[k][i] = torch.from_numpy(
                            np.concatenate(
                            [input[k][i], padding], axis=0
                        )
                    )
        
        collated_input['times'] = torch.from_numpy(np.stack([self.pad_to_max_size(s['times'], target_seq_size) for s in samples]))

        out['net_input'] = collated_input

        if 'labels' in samples[0].keys():
            label_dict = dict()

            for k in samples[0]['labels'].keys():
                label_dict[k] = torch.stack([s['labels'][k] for s in samples])
            
            out['labels'] = label_dict

        return out

    def pad_to_max_size(self, sample, max_len):
        if len(sample) < max_len:
            sample = np.concatenate(
                [sample, np.zeros(max_len - len(sample), dtype=np.int16)]
            )
        return sample

    def __getitem__(self, index):

        fname = str(int(self.hit_idcs[index])) + '.pkl' 
        data = pd.read_pickle(os.path.join(self.data_dir, fname))

        word_len = []
        for event in data[self.structure][:, 0, :]:
            if 0 in event:
                word_len.append(np.where(event==0)[0][0])
            else:
                word_len.append(len(word_len))
        word_len = max(word_len)

        pack = {
            'input_ids': data[self.structure][:, 0, :][:, :word_len],
            'type_ids': data[self.structure][:, 1, :][:, :word_len],
            'dpe_ids': data[self.structure][:, 2, :][:, :word_len],
            'times': data['time'],
        } 
        
        # Labels
        labels = dict()
        for task in self.pred_tasks:
            task_name = task.name
            task_prop = task.property
            task_class = task.num_classes
            labels[task_name] = self.fold_file[self.fold_file[self.stay_id] == self.hit_idcs[index]][task_name].values[0]
            if task_prop == "binary":
                labels[task_name] = torch.tensor(labels[task_name], dtype=torch.float32)
            elif task_prop == "multi-label": 
                if task_name == 'diagnosis':
                    if not isinstance(labels[task_name], str): # nan no loss
                        labels[task_name] = torch.zeros(task_class).to(torch.float32)
                    else:
                        labels[task_name] = eval(labels[task_name]) #[3,5,2,1]
                        labels[task_name] = F.one_hot(torch.tensor(labels[task_name]).to(torch.int64), num_classes=task_class).sum(dim=0).bool().to(torch.float32)
            elif task_prop == "multi-class":
                # Missing values are filled with -1 or Nan
                if labels[task_name] == -1 or np.isnan(labels[task_name]):
                    labels[task_name] = torch.zeros(task_class).to(torch.float32)
                else:
                    labels[task_name] = F.one_hot(torch.tensor(labels[task_name]).to(torch.int64), num_classes=task_class).to(torch.float32)
        
        pack['labels'] = labels

        return pack