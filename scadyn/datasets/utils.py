import os
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.fileio import load

class SimpleClipRecord(object):
    def __init__(self, clipdata=None, dur=3):
        self.data = {}
        self.clipname = clipdata['clip_id']
        self.video, start_meta, end_meta = self.clipname.split('_')
        self.start_frame = int(start_meta[2:])
        self.end_frame   = int(end_meta[2:])

        # check if the duration is 3 seconds
        assert self.end_frame - self.start_frame == dur, f"Duration is not {dur} seconds"
        assert clipdata['futr_end_t'] - clipdata['futr_start_t'] == dur, f"Future duration is not {dur} seconds"

        self.update("cur_v", clipdata['cur_verb'])
        self.update("cur_ivt", clipdata['cur_triplet'])
        self.update("cur_p", clipdata['cur_phase'])
        self.update("futr_v", clipdata['futr_verb'])
        self.update("futr_ivt", clipdata['futr_triplet'])
        self.update("futr_p", clipdata['futr_phase'])

        # next clip
        self.update("next_clip_name", clipdata['futr_clip_name'])

        # check if rsd attribute exists
        if "rsd" in clipdata:
            self.update("remaining_surgery_duration", clipdata["rsd"])

        if "phase_transition" in clipdata:
            self.update("phase_transition", clipdata["phase_transition"])

        if "instrument_transition" in clipdata:
            self.update("instrument_transition", clipdata["instrument_transition"])

        # TODO: add remaining surgery duration and phase transition duration and precompute in the json file.
        # if "remaining_surgery_duration" in clipdata:
        #     self.update("remaining_surgery_duration", clipdata["remaining_surgery_duration"])

        # if "phase_transition_duration" in clipdata:
        #     self.update("phase_transition", clipdata["phase_transition_duration"])

    def update(self, key, value):
        self.data[key] = value

    @property
    def clipinfo(self):
        return self.clipname
    
    @property
    def cur_phase(self):
        return self.data['cur_p']
    
    @property
    def cur_verb(self):
        return self.data['cur_v']
    
    @property
    def cur_triplet(self):
        return self.data['cur_ivt']
    
    @property
    def futr_verb(self):
        return self.data['futr_v'] # future_triplet
    
    @property
    def futr_triplet(self):
        return self.data['futr_ivt']
    
    @property
    def futr_phase(self):
        return self.data['futr_p']
    
    @property
    def next_clip_name(self):
        return self.data["next_clip_name"] # name of the next clip
    
    @property
    def rsd(self):
        if "remaining_surgery_duration" in self.data:
            return self.data["remaining_surgery_duration"] # in seconds
        else:
            return None

    @property
    def phtr(self):
        if "phase_transition" in self.data:
            return self.data["phase_transition"] # in seconds
        else:
            return None
        
    @property
    def iantr(self):
        if "instrument_transition" in self.data:
            return self.data["instrument_transition"] # in seconds
        else:
            return None
        
        

class ClipRecord(object):
    def __init__(self, clipname, clipdata=None, update=False, dur=10):
        self.data = {}
        self.clipname = clipname
        if len(self.clipname.split('_')) == 3:
            video_meta, start_meta, end_meta = self.clipname.split('_')
            start_frame = int(start_meta[2:])
            end_frame   = int(end_meta[2:])
        else:
            video_meta, start_meta = self.clipname.split('_')
            start_frame = int(start_meta[1:])*dur
            end_frame   = int(start_frame + dur)
        self.video = video_meta
        self.start_frame = start_frame
        self.end_frame = end_frame

        if update and clipdata!=None:
            self.update("tool", clipdata['i'])
            self.update("verb", clipdata['v'])
            self.update("target", clipdata['t'])
            self.update("tool-action", clipdata['iv'])
            self.update("tool-tissue", clipdata['it'])
            self.update("triplet", clipdata['ivt'])
            self.update("phase", clipdata['phase'])

            if "future_frames" in clipdata:
                self.update("future-triplet", clipdata['future_frames'])
                self.update("future-frame-ids", clipdata['future_frame_ids'])
                self.update("future-horizon", len(clipdata['future_frame_ids']))

            if 'next_clip' in clipdata:
                self.update("next_clip_name", clipdata['next_clip']['clipname'])
                self.update("next_clip_triplet", clipdata['next_clip']['labels'])

            if "remaining_surgery_duration" in clipdata:
                self.update("remaining_surgery_duration", clipdata["remaining_surgery_duration"])

            if "phase_transition_duration" in clipdata:
                self.update("phase_transition", clipdata["phase_transition_duration"])
            
            if "next_phase_label" in clipdata:
                self.update("next_phase_label", clipdata["next_phase_label"])

    def update(self, key, value):
        self.data[key] = value

    @property
    def clipinfo(self):
        return self.clipname
    
    @property
    def tool(self):
        return self.data['tool']
    
    @property
    def phase(self):
        return self.data['phase']
    
    @property
    def verb(self):
        return self.data['verb']

    @property
    def target(self):
        return self.data['target']
    
    @property
    def triplet(self):
        return self.data['triplet']

    @property
    def tool_target(self):
        return self.data['tool-tissue'] # tool-tissue

    @property
    def tool_verb(self):
        return self.data['tool-action'] # tool-verb
    
    @property
    def future_triplet(self):
        return self.data['future-triplet'] # future_triplet
    
    @property
    def future_frame_ids(self):
        return self.data["future-frame-ids"] # frame ids of the k future frames
    
    @property
    def future_horizon(self):
        return self.data["future-horizon"] # number of future frames
    
    @property
    def next_clip_name(self):
        return self.data["next_clip_name"] # name of the next clip
    
    @property
    def next_clip_triplet(self):
        return self.data["next_clip_triplet"] # triplet of the next clip
    
    @property
    def rsd(self):
        if "remaining_surgery_duration" in self.data:
            return self.data["remaining_surgery_duration"] # in seconds
        else:
            return None

    @property
    def phase_transition(self):
        if "phase_transition" in self.data:
            return self.data["phase_transition"] # in seconds
        else:
            return None
        
    @property
    def next_phase_label(self):
        if "next_phase_label" in self.data:
            return self.data["next_phase_label"]
        else:
            return None

cmap = {
        'i':{'idx':0, 'nc':6}, 
        'v':{'idx':1,'nc':10},
        't':{'idx':2,'nc':15},
        'iv':{'idx':3,'nc':26},
        'it':{'idx':4,'nc':59},
        'ivt':{'idx':5, 'nc':100},
        'phase': {'idx':6, 'nc':6},
        'it_adj': {'idx':7, 'nc':90}
}
components = ['i', 'v', 't', 'it', 'iv', 'ivt']
mapfile = 'data/state_change/triplet_mappers.json'
# mapping from triplet and components to index
i2namemap = load(mapfile) 
reverse_map = {}
for k,v in i2namemap.items():
    if k not in reverse_map:
        reverse_map[k] = {q:p for p,q in i2namemap[k].items()}

def prosta_to_components(label_onehot, component=['v']):
    cmap = {
            'i':{'idx':0, 'nc':7}, 
            'v':{'idx':1,'nc':10},
            't':{'idx':2,'nc':10},
            'ivt':{'idx':3, 'nc':89},
    }
    # components = ['i', 'v', 't', 'ivt']
    mapfile = 'data/state_change/prosta_triplet_mappers.json'
    # mapping from triplet and components to index
    prosta_i2namemap = load(mapfile) 
    reverse_map = {}
    for k,v in prosta_i2namemap.items():
        if k not in reverse_map:
            reverse_map[k] = {q:p for p,q in prosta_i2namemap[k].items()}

    all_component_labels = []
    for c in component:
        label_list = []
        for k in label_onehot:
            label = [0]*cmap[c]['nc']
            ivt_ids = list(np.where(k)[0])
            if len(ivt_ids) != 0:
                for ivt_id in ivt_ids:
                    i, v, t = prosta_i2namemap['ivt'][str(ivt_id)].split(",")
                    if c == 'it':
                        key = f"{i},{t}"
                    elif c == 'iv':
                        key = f"{i},{v}"
                    elif c == 'v':
                        key = f"{v}"
                    elif c == 't':
                        key = f"{t}"
                    elif c == 'i':
                        key = f"{i}"
                    elif c == 'ivt':
                        key = f"{i},{v},{t}"
                    else:
                        raise ValueError("Illegal component name")            
                    idx = reverse_map[c][key]
                    label[int(idx)] = 1

            label_list.append(label)
        all_component_labels.append(np.array(label_list))
    
    # return all_component_labels if len(all_component_labels) > 1 else all_component_labels[0]
    if len(component) == 1:
        return all_component_labels[0]
    else:
        return all_component_labels

def to_components(label_onehot, component=['v']):
    all_component_labels = []
    for c in component:
        label_list = []
        for k in label_onehot:
            label = [0]*cmap[c]['nc']
            ivt_ids = list(np.where(k)[0])
            if len(ivt_ids) != 0:
                for ivt_id in ivt_ids:
                    i, v, t = i2namemap['ivt'][str(ivt_id)].split(",")
                    if c == 'it':
                        key = f"{i},{t}"
                    elif c == 'iv':
                        key = f"{i},{v}"
                    elif c == 'v':
                        key = f"{v}"
                    elif c == 't':
                        key = f"{t}"
                    elif c == 'i':
                        key = f"{i}"
                    elif c == 'ivt':
                        key = f"{i},{v},{t}"
                    else:
                        raise ValueError("Illegal component name")            
                    idx = reverse_map[c][key]
                    label[int(idx)] = 1

            label_list.append(label)
        all_component_labels.append(np.array(label_list))
    
    # return all_component_labels if len(all_component_labels) > 1 else all_component_labels[0]
    if len(component) == 1:
        return all_component_labels[0]
    else:
        return all_component_labels

def gen_state_change_labels(labels, fgbg=False):
    # to find difference between labels at time t and t+1
    num_futrs = labels.shape[0]-1 #?
    cur_labels = labels
    fut_labels = labels[1:]

    state_change_ids_tracks = []
    for i in range(num_futrs):
        cur_label = cur_labels[i]
        fut_label = fut_labels[i]
        state_change_ids = [0]*len(cur_label)
        for k, (c, f) in enumerate(zip(cur_label, fut_label)):
            if c == 0 and f == 1:
                state_change_ids[k] = 2 if not fgbg else 1
            elif c == 1 and f == 0:
                state_change_ids[k] = 1 if not fgbg else 0
            elif c == 1 and f == 1:
                state_change_ids[k] = 0 if not fgbg else 1
            else:
                state_change_ids[k] = 3 if not fgbg else 0
        state_change_ids_tracks.append(state_change_ids)
    state_change_ids_tracks = torch.tensor(state_change_ids_tracks)

    return state_change_ids_tracks

# prepare state change of component labels based on the cur and future labels 
def generate_state_change_labels(labels, fgbg=False):
    num_futrs = labels.shape[0]-1

    # to find difference between labels at time t and t+1
    cur_labels = labels
    fut_labels = labels[1:]

    state_change_ids_tracks = []
    for i in range(num_futrs):
        cur_label = cur_labels[i]
        fut_label = fut_labels[i]
        state_change_ids = [0]*len(cur_label)
        for k, (c, f) in enumerate(zip(cur_label, fut_label)):
            if c == 0 and f == 1:
                state_change_ids[k] = 2 if not fgbg else 1
            elif c == 1 and f == 0:
                state_change_ids[k] = 1 if not fgbg else 0
            elif c == 1 and f == 1:
                state_change_ids[k] = 0 if not fgbg else 1
            else:
                state_change_ids[k] = 3 if not fgbg else 0

        state_change_ids_tracks.append(state_change_ids)
    state_change_ids_tracks = torch.tensor(state_change_ids_tracks)

    # get onehot labels for state change
    num_classes = 2 if fgbg else 4
    state_change_onehot = F.one_hot(state_change_ids_tracks, num_classes=num_classes)
    return state_change_onehot,state_change_ids_tracks


class StateTracker:
    def __init__(self, num_classes=4, name='cdo'):
        self.data = {k:0 for k in range(num_classes)}
        self.num_instances = 0
        self.num_classes = num_classes
        self.name = name

    def update_per_class(self, state):
        for k in state:
            self.data[k] += 1
            self.num_instances += 1

    def update_per_presence(self, state):
        for k in list(self.data.keys()):
            if k in state:
                self.data[k] += 1
        self.num_instances += 1

    def info(self):
        print(f"{self.name} State Tracker %ages")
        for k in self.data:
            print(f"State {k}: {self.data[k]}/{self.num_instances}={self.data[k]/self.num_instances*100:.2f}%")