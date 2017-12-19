import json
import re
import os

import numpy as np
import pandas as pd
import paprotka.io as pio


def get_root(config='paths.json'):
    with open(config) as opened:
        paths = json.load(opened)
    return paths['reddots_root']


def read_files(*paths, encoding='utf-8'):
    for path in paths:
        with open(path, encoding=encoding) as opened:
            yield from opened


recording_regex = re.compile(r'([mf])(\d+)/(\d+)_[mf]\d+_(\d+)')
def parse_recording(value):
    match = recording_regex.match(value)
    if match:
        return match.groups()


speaker_sentence_regex = re.compile(r'([mf])(\d+)_(\d+)')
def parse_speaker_sentence(value):
    match = speaker_sentence_regex.match(value)
    if match:
        return match.groups()


timestamp_format='%Y%m%d%H%M%S%f'
def parse_timestamp(timestamp):
    return pd.to_datetime(timestamp, format=timestamp_format)


def parse_gender(flag):
    return flag == 'm'


def parse_flag(flag):
    return flag == 'Y'


def load_trials(*paths):
    columns = ([], [], [], [], [], [], [], [], [], [])
    
    for line in read_files(*paths):
        speaker_sentence, recording, tc, tw, ic, iw = line.strip().split(',')
        
        expected_gender, expected_speaker_id, expected_sentence_id = parse_speaker_sentence(speaker_sentence)
        trial_gender, trial_speaker_id, trial_timestamp, trial_sentence_id = parse_recording(recording)
        
        expected_is_male = parse_gender(expected_gender)
        trial_is_male = parse_gender(trial_gender)
        
        target_person = parse_flag(tc) or parse_flag(tw)
        correct_sentence = parse_flag(tc) or parse_flag(ic)
        
        trial_timestamp = parse_timestamp(trial_timestamp)
        pcm_path = recording + '.pcm'
        
        record = (expected_is_male, expected_speaker_id, expected_sentence_id, 
                  trial_is_male, trial_speaker_id, trial_timestamp, trial_sentence_id,
                  pcm_path, target_person, correct_sentence)
        
        for value, column in zip(record, columns):
            column.append(value)
        
    data = {'expected_is_male':     pd.Series(columns[0], dtype=np.bool), 
            'expected_speaker_id':  pd.Series(columns[1], dtype=np.int16), 
            'expected_sentence_id': pd.Series(columns[2], dtype=np.int16), 
            'trial_is_male':        pd.Series(columns[3], dtype=np.bool), 
            'trial_speaker_id':     pd.Series(columns[4], dtype=np.int16), 
            'trial_timestamp':      pd.Series(columns[5], dtype='datetime64[ns]'), 
            'trial_sentence_id':    pd.Series(columns[6], dtype=np.int16),
            'pcm_path':             pd.Series(columns[7], dtype=str), 
            'target_person':        pd.Series(columns[8], dtype=np.bool), 
            'correct_sentence':     pd.Series(columns[9], dtype=np.bool)}
    return pd.DataFrame.from_dict(data)
        

def load_enrollments(*paths):
    columns = ([], [], [], [], [])

    for line in read_files(*paths):
        speaker_sentence, recordings = line.strip().split(' ', 1)
        for recording in recordings.split(','):
            gender, speaker_id, timestamp, sentence_id = parse_recording(recording)
            
            is_male = parse_gender(gender)
            timestamp = parse_timestamp(timestamp)
            pcm_path = recording + '.pcm'
            
            record = (is_male, speaker_id, timestamp, sentence_id, pcm_path)

            for value, column in zip(record, columns):
                column.append(value)

    data = {'is_male':     pd.Series(columns[0], dtype=np.bool), 
            'speaker_id':  pd.Series(columns[1], dtype=np.int16), 
            'timestamp':   pd.Series(columns[2], dtype='datetime64[ns]'), 
            'sentence_id': pd.Series(columns[3], dtype=np.int16),
            'pcm_path':    pd.Series(columns[4], dtype=str)}
    return pd.DataFrame.from_dict(data)


def load_pcm(root, path):
    full_path = os.path.join(root, 'pcm', path)
    return np.fromfile(full_path, np.int16)


def load_pcm_snd(root, path):
    full_path = os.path.join(root, 'pcm', path)
    return pio.load_pcm(full_path, dtype=np.int16, rate=16000)


def load_npy(root, source, path):
    full_path = os.path.join(root, source, path)
    return np.load(full_path)


def transform_all_recordings(root, source, target, function):
    for person_dir in os.listdir(os.path.join(root, source)):
        source_dir = os.path.join(root, source, person_dir)
        target_dir = os.path.join(root, target, person_dir)
        
        os.makedirs(target_dir)
        for source_file in os.listdir(source_dir):
            target_file = source_file if source != 'pcm' else source_file[:-3] + 'npy'
            
            source_path = os.path.join(source_dir, source_file)
            target_path = os.path.join(target_dir, source_file)
            
            if source == 'pcm':
                source_data = np.fromfile(source_path, np.int16)
            else: 
                source_data = np.load(source_path)
                
            result = function(source_data)
            
            np.save(target_path, result)

