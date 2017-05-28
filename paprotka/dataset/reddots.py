import os
import numpy as np
import paprotka.io as pio


def read_paths(read_file, *paths):
    for path in paths:
        with open(path) as opened:
            read_file(opened)


def load_enrollments(*paths):
    people, pcm_triples = [], []

    def read_file(opened):
        for line in opened.readlines():
            person, pcm_paths = line.strip().split(' ', 1)
            pcm_triple = pcm_paths.split(',')
            people.append(person)
            pcm_triples.append(pcm_triple)

    read_paths(read_file, *paths)
    return np.array(people), np.array(pcm_triples)


def load_trials(*paths):
    people, pcm_paths, truths = [], [], []

    def read_file(opened):
        for line in opened.readlines():
            person, pcm_path, *targets = line.strip().split(',')
            truth_row = [target == 'Y' for target in targets]
            # same prsn + same sent / same prsn + diff sent / diff prsn + same sent / diff prsn diff sent
            people.append(person)
            pcm_paths.append(pcm_path)
            truths.append(truth_row)

    read_paths(read_file, *paths)
    return np.array(people), np.array(pcm_paths), np.array(truths, dtype=np.bool)


def load_pcm(root, path):
    full_path = os.path.join(root, '{}.pcm'.format(path))
    return pio.load_pcm(full_path, dtype=np.int16, rate=16000)


def filter_trials(trials, booleans):
    people, pcm_paths, truths = trials
    return people[booleans], pcm_paths[booleans], truths[booleans]


def filter_enrollments(enrollments, booleans):
    people, path_triples = enrollments
    return people[booleans], path_triples[booleans]


def filter_correct_sentence(trials):
    _, _, truths = trials
    correct_sentence = truths[:, [0, 2]].any(axis=1)
    return filter_trials(trials, correct_sentence)


def filter_correct_speaker(trials):
    _, _, truths = trials
    correct_sentence = truths[:, [0, 1]].any(axis=1)
    return filter_trials(trials, correct_sentence)


def group_same_sentence(enrollments, trials):
    e_people = enrollments[0]
    t_people = trials[0]
    e_sentence_codes = np.array([person.rsplit('_', 1)[1] for person in e_people])
    t_sentence_codes = np.array([person.rsplit('_', 1)[1] for person in t_people])

    all_sentence_codes = np.hstack((e_sentence_codes, t_sentence_codes))
    unique_codes = np.unique(all_sentence_codes)

    def make_group(code):
        filtered_enrollments = filter_enrollments(enrollments, e_sentence_codes == code)
        filtered_trials = filter_trials(trials, t_sentence_codes == code)
        return filtered_enrollments, filtered_trials

    return {int(code): make_group(code) for code in unique_codes}
