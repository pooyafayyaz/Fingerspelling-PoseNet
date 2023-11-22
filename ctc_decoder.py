import math
import numpy as np
import torch
import torch.utils.data as tud
from tqdm.auto import tqdm
import warnings


def beam_search(
    model, 
    X,
    poses,
    predictions = 20,
    beam_width = 5,
    batch_size = 128, 
    progress_bar = 0
):
    """
    Implements Beam Search to extend the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width. 

    progress_bar: bool
        Shows a tqdm progress bar, useful for tracking progress with large tensors.

    Returns
    -------
    X: LongTensor of shape (examples, length + predictions)
        The sequences extended with the decoding process.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """
    with torch.no_grad():
        # The next command can be a memory bottleneck, but can be controlled with the batch 
        # size of the predict method.

        next_probabilities = model.forward(poses,X)[0][:,-1, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1)\
            .topk(k = beam_width, axis = -1)
        X = X.repeat((beam_width, 1, 1)).transpose(0, 1)\
            .flatten(end_dim = -2)
        next_chars = idx.reshape(-1, 1)
        X = torch.cat((X, next_chars), axis = -1)
        
        poses = torch.cat((poses,poses, poses, poses, poses), axis = 0)
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
            dataset = tud.TensorDataset(X)
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            next_probabilities = []
            iterator = iter(loader)
            if progress_bar > 1:
                iterator = tqdm(iterator)
            for (x,) in iterator:
                next_probabilities.append(
                    model.forward(poses, x)[0][:,-1, :].log_softmax(-1)
                )
            next_probabilities = torch.cat(next_probabilities, axis = 0)
            next_probabilities = next_probabilities.reshape(
                (-1, beam_width, next_probabilities.shape[-1])
            )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities = probabilities.flatten(start_dim = 1)
            probabilities, idx = probabilities.topk(
                k = beam_width, 
                axis = -1
            )
            next_chars = torch.remainder(idx, vocabulary_size).flatten()\
                .unsqueeze(-1)
            best_candidates = (idx / vocabulary_size).long()
            best_candidates += torch.arange(
                X.shape[0] // beam_width, 
                device = X.device
            ).unsqueeze(-1) * beam_width
            X = X[best_candidates].flatten(end_dim = -2)
            X = torch.cat((X, next_chars), axis = 1)
        return X.reshape(-1, beam_width, X.shape[-1]), probabilities

class Decoder(object):
    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.char_to_int = dict([(c, i) for (i, c) in enumerate(labels)])

        print(self.int_to_char)
        # print(self.char_to_int)

        self.blank_index = blank_index
        space_index = len(labels)
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def greedy_decode(self, prob, digit=False):
        # prob: [seq_len, num_labels+1], numpy array
        indexes = np.argmax(prob, axis=1)
        string = []
        prev_index = -1
        for i in range(len(indexes)):
            if indexes[i] == self.blank_index:
                prev_index = -1
                continue
            elif indexes[i] == prev_index:
                continue
            else:
                if digit is False:
                    
                    if len(string)>1 and self.int_to_char[indexes[i]]==string[-1]:
                        continue
                    string.append(self.int_to_char[indexes[i]])
                else:
                    string.append(indexes[i])
                prev_index = indexes[i]
        return string

    def beam_decode(self, prob, beam_size, beta=0.0, gamma=0.0, scorer=None, digit=False):
        # prob: [seq_len, num_labels+1], numpy array
        # beta: lm coef, gamma: insertion coef
        seqlen = len(prob)
        beam_idx = np.argsort(prob[0, :])[-beam_size:].tolist()

        beam_prob = list(map(lambda x: math.log(prob[0, x]), beam_idx))
        beam_idx = list(map(lambda x: [x], beam_idx))
        for t in range(1, seqlen):
            topk_idx = np.argsort(prob[t, :])[-beam_size:].tolist()
            topk_prob = list(map(lambda x: prob[t, x], topk_idx))
            aug_beam_prob, aug_beam_idx = [], []
            for b in range(beam_size*beam_size):
                aug_beam_prob.append(beam_prob[int(b/beam_size)])
                aug_beam_idx.append(list(beam_idx[int(b/beam_size)]))
            # allocate
            for b in range(beam_size*beam_size):
                i, j = int(b/beam_size), int(b % beam_size)
                aug_beam_idx[b].append(topk_idx[j])
                aug_beam_prob[b] = aug_beam_prob[b]+math.log(topk_prob[j])
            # merge
            merge_beam_idx, merge_beam_prob = [], []
            for b in range(beam_size*beam_size):
                if aug_beam_idx[b][-1] == aug_beam_idx[b][-2]:
                    beam, beam_prob = aug_beam_idx[b][:-1], aug_beam_prob[b]
                elif aug_beam_idx[b][-2] == self.blank_index:
                    beam, beam_prob = aug_beam_idx[b][:-2]+[aug_beam_idx[b][-1]], aug_beam_prob[b]
                else:
                    beam, beam_prob = aug_beam_idx[b], aug_beam_prob[b]
                beam_str = list(map(lambda x: self.int_to_char[x], beam))
                if beam_str not in merge_beam_idx:
                    merge_beam_idx.append(beam_str)
                    merge_beam_prob.append(beam_prob)
                else:
                    idx = merge_beam_idx.index(beam_str)
                    merge_beam_prob[idx] = np.logaddexp(merge_beam_prob[idx], beam_prob)

            if scorer is not None:
                merge_beam_prob_lm, ins_bonus, strings = [], [], []
                for b in range(len(merge_beam_prob)):
                    if merge_beam_idx[b][-1] == self.int_to_char[self.blank_index]:
                        strings.append(merge_beam_idx[b][:-1])
                        ins_bonus.append(len(merge_beam_idx[b][:-1]))
                    else:
                        strings.append(merge_beam_idx[b])
                        ins_bonus.append(len(merge_beam_idx[b]))
                lm_scores = scorer.get_score_fast(strings)
                for b in range(len(merge_beam_prob)):
                    total_score = merge_beam_prob[b]+beta*lm_scores[b]+gamma*ins_bonus[b]
                    merge_beam_prob_lm.append(total_score)

            if scorer is None:
                ntopk_idx = np.argsort(np.array(merge_beam_prob))[-beam_size:].tolist()
            else:
                ntopk_idx = np.argsort(np.array(merge_beam_prob_lm))[-beam_size:].tolist()
            beam_idx = list(map(lambda x: merge_beam_idx[x], ntopk_idx))
            for b in range(len(beam_idx)):
                beam_idx[b] = list(map(lambda x: self.char_to_int[x], beam_idx[b]))
            beam_prob = list(map(lambda x: merge_beam_prob[x], ntopk_idx))
        if self.blank_index in beam_idx[-1]:
            pred = beam_idx[-1][:-1]
        else:
            pred = beam_idx[-1]
        if digit is False:
            pred = list(map(lambda x: self.int_to_char[x], pred))
        return pred

    # def get_trans_score(self, decoder , strings):

    def beam_decode_trans(self, prob, beam_size, decoder, poses , beta=0.0, gamma=0.0, digit=False):
        # prob: [seq_len, num_labels+1], numpy array
        # beta: lm coef, gamma: insertion coef
        seqlen = len(prob)
        beam_idx = np.argsort(prob[0, :])[-beam_size:].tolist()

        beam_prob = list(map(lambda x: math.log(prob[0, x]), beam_idx))
        beam_idx = list(map(lambda x: [x], beam_idx))
        for t in range(1, seqlen):
            topk_idx = np.argsort(prob[t, :])[-beam_size:].tolist()
            topk_prob = list(map(lambda x: prob[t, x], topk_idx))
            aug_beam_prob, aug_beam_idx = [], []
            for b in range(beam_size*beam_size):
                aug_beam_prob.append(beam_prob[int(b/beam_size)])
                aug_beam_idx.append(list(beam_idx[int(b/beam_size)]))
            # allocate
            for b in range(beam_size*beam_size):
                i, j = int(b/beam_size), int(b % beam_size)
                aug_beam_idx[b].append(topk_idx[j])
                aug_beam_prob[b] = aug_beam_prob[b]+math.log(topk_prob[j])
            # merge
            merge_beam_idx, merge_beam_prob = [], []
            for b in range(beam_size*beam_size):
                if aug_beam_idx[b][-1] == aug_beam_idx[b][-2]:
                    beam, beam_prob = aug_beam_idx[b][:-1], aug_beam_prob[b]
                elif aug_beam_idx[b][-2] == self.blank_index:
                    beam, beam_prob = aug_beam_idx[b][:-2]+[aug_beam_idx[b][-1]], aug_beam_prob[b]
                else:
                    beam, beam_prob = aug_beam_idx[b], aug_beam_prob[b]
                beam_str = list(map(lambda x: self.int_to_char[x], beam))
                if beam_str not in merge_beam_idx:
                    merge_beam_idx.append(beam_str)
                    merge_beam_prob.append(beam_prob)
                else:
                    idx = merge_beam_idx.index(beam_str)
                    merge_beam_prob[idx] = np.logaddexp(merge_beam_prob[idx], beam_prob)

            merge_beam_prob_lm, ins_bonus, strings = [], [], []
            for b in range(len(merge_beam_prob)):
                if merge_beam_idx[b][-1] == self.int_to_char[self.blank_index]:
                    strings.append(merge_beam_idx[b][:-1])
                    ins_bonus.append(len(merge_beam_idx[b][:-1]))
                else:
                    strings.append(merge_beam_idx[b])
                    ins_bonus.append(len(merge_beam_idx[b]))
            
            lm_scores = decoder.return_scores(poses,strings,self.char_to_int)

            for b in range(len(merge_beam_prob)):
                total_score = merge_beam_prob[b]+beta*lm_scores[b]+gamma*ins_bonus[b]
                merge_beam_prob_lm.append(total_score)

            ntopk_idx = np.argsort(np.array(merge_beam_prob_lm))[-beam_size:].tolist()
            
            beam_idx = list(map(lambda x: merge_beam_idx[x], ntopk_idx))
            for b in range(len(beam_idx)):
                beam_idx[b] = list(map(lambda x: self.char_to_int[x], beam_idx[b]))
            beam_prob = list(map(lambda x: merge_beam_prob[x], ntopk_idx))
        if self.blank_index in beam_idx[-1]:
            pred = beam_idx[-1][:-1]
        else:
            pred = beam_idx[-1]
        if digit is False:
            pred = list(map(lambda x: self.int_to_char[x], pred))
        return pred
