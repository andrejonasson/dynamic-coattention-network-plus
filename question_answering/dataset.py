import numpy as np
from preprocessing.qa_data import PAD_ID

class SquadDataset:
    def __init__(self, question_file, paragraph_file, answer_file, max_question_length, max_paragraph_length):
        self.question_file = question_file
        self.paragraph_file = paragraph_file
        self.answer_file = answer_file
        self.max_question_length = max_question_length
        self.max_paragraph_length = max_paragraph_length
        self.length = -1

        # read files into memory to save I/O cost
        # also checks that lengths match
        self._read_into_memory()
        self.epoch_sampled = 0
        self.epoch = 1

    def get_batch(self, batch_size, replace=True):
        if replace:
            batch_idx = np.random.choice(self.length, batch_size, replace)
            return self[batch_idx]

        if self.epoch_sampled == 0 and self.epoch == 1:
            self.shuffle()

        if self.epoch_sampled == self.length:
            self.shuffle()
            self.epoch_sampled = 0
            self.epoch += 1
        
        tail_batch = []
        if self.epoch_sampled + batch_size > self.length:
            tail_batch = self[self.epoch_sampled:]
            batch_size -= len(tail_batch[0])
            self.shuffle()
            self.epoch_sampled = 0
            self.epoch += 1
        
        batch = self[self.epoch_sampled:self.epoch_sampled+batch_size]
        self.epoch_sampled += batch_size
        if tail_batch:
            batch = self.join(tail_batch, batch)
        return batch


    def shuffle(self):
        """ https://stackoverflow.com/questions/11765061/better-way-to-shuffle-two-related-lists """
        from random import shuffle
        question_shuf = []
        paragraph_shuf = []
        answer_shuf = []
        index_shuf = list(range(self.length))
        shuffle(index_shuf)

        for i in index_shuf:
            question_shuf.append(self.question[i])
            paragraph_shuf.append(self.paragraph[i])
            answer_shuf.append(self.answer[i])

        self.question, self.paragraph, self.answer = tuple(question_shuf), tuple(paragraph_shuf), tuple(answer_shuf)


    @staticmethod
    def join(first, second):
        a = []
        for i, j in zip(first, second):
            a.append(i+j)
        return tuple(a)


    def __getitem__(self, arg):
        from collections.abc import Iterable

        if isinstance(arg, int):
            arg = [arg]

        if isinstance(arg, Iterable):
            questions, question_lengths = pad_sequences([self.question[i] for i in arg], self.max_question_length)
            paragraphs, paragraph_lengths = pad_sequences([self.paragraph[i] for i in arg], self.max_paragraph_length)
            answers = [self.answer[i] for i in arg]

        if isinstance(arg, slice):
            questions, question_lengths = pad_sequences(self.question[arg], self.max_question_length)
            paragraphs, paragraph_lengths = pad_sequences(self.paragraph[arg], self.max_paragraph_length)
            answers = self.answer[arg]

        return (questions, paragraphs, question_lengths, paragraph_lengths, answers)


    @staticmethod
    def read_file(file):
        def process_line(line):
            data = line.strip().split(' ')
            data = [int(pt) for pt in data]
            return data

        with open(file) as f:
            dataset = [process_line(line) for line in f]
        return dataset


    def _read_into_memory(self):
        question = self.read_file(self.question_file)
        paragraph = self.read_file(self.paragraph_file)
        answer = self.read_file(self.answer_file)

        # remove all questions that are cut by its max length
        # and answers that start or end after paragraph max length
        self.question, self.paragraph, self.answer = zip(*
            [
                (q, p, a) 
                for q, p, a in zip(question, paragraph, answer) 
                if len(p) < self.max_paragraph_length
                and len(q) < self.max_question_length   # max(a) < self.max_paragraph_length Questionable
            ]
        )

        assert len(self.question) == len(self.answer) == len(self.paragraph)
        self.length = len(self.question)


def pad_sequence(sequence, max_length):
        """ Pads data of format `(sequence, labels)` to `max_length` sequence length
        and returns a triplet `(sequence_, labels_, mask)`. If
        the length of the sequence is longer than `max_length` then it 
        is truncated to `max_length`.
        """

        # create padding vectors
        sequence_padding = PAD_ID
        
        pad_length = max([0, max_length - len(sequence)])
        padded_sequence = sequence[:max_length]
        padded_sequence.extend([sequence_padding]*pad_length)
        length = min([len(sequence), max_length])

        return padded_sequence, length


def pad_sequences(sequences, max_length):
    padded_sequences, lengths = zip(*[pad_sequence(sequence, max_length) for sequence in sequences])
    return padded_sequences, lengths


if __name__ == '__main__':
    from utils import get_data_paths
    import os
    data_dir = os.path.join("..", "data", "squad")
    dataset = SquadDataset(*get_data_paths(data_dir, name='train'), 10, 100)
    print(dataset.get_batch(2))
    print(dataset[:2])
    print(dataset.question)
