import numpy as np
import pdb

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Union
from torch import Tensor


def spaced_elements(array, num_elems=4):
    return [x[len(x)//2] for x in np.array_split(np.array(array), num_elems)]


class AbstractPlayerIterator(ABC):
    def __init__(self, ground_input, windows, random=False):
        self._assert_input_compatibility(ground_input)
        if isinstance(ground_input, Tuple):
            ground_input, feats = ground_input
            self.feat_shape = feats.shape[1:]
        self.windows = windows
        self.input_shape = ground_input.shape[1:]
        self.random = random
        self.n_players = self._get_number_of_players_from_shape()
        self.permutation = np.array(range(self.n_players), 'int32')
        if random is True:
            self.permutation = np.random.permutation(self.permutation)
        self.i = 0
        self.kn = self.n_players
        self.ks = spaced_elements(range(self.n_players), self.kn)
        self.coalition_size = 1

    def set_n_steps(self, steps):
        self.kn = steps
        self.ks = spaced_elements(range(self.n_players), self.kn)

    def get_number_of_players(self):
        return self.n_players

    def get_explanation_shape(self):
        return self.input_shape

    def get_coalition_size(self):
        return 1

    def get_steps_list(self):
        return self.ks

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.n_players:
            raise StopIteration
        m = self._get_masks_for_index(self.i)
        self.i = self.i + 1
        return m

    @abstractmethod
    def _assert_input_compatibility(self, ground_input):
        pass

    @abstractmethod
    def _get_masks_for_index(self, i):
        pass

    @abstractmethod
    def _get_number_of_players_from_shape(self):
        pass


class DefaultPlayerIterator(AbstractPlayerIterator):

    def _assert_input_compatibility(self, ground_input):
        assert len(ground_input.shape) > 1, 'DefaultPlayerIterator requires an input with 2 or more dimensions'

    def _get_number_of_players_from_shape(self):
        return int(np.prod(self.input_shape))

    def _get_masks_for_index(self, i):
        mask = np.zeros(self.n_players, dtype='int32')
        mask[self.permutation[i]] = 1
        return mask.reshape(self.input_shape), mask.reshape(self.input_shape)


class ImagePlayerIterator(AbstractPlayerIterator):

    def __init__(self, 
                ground_input,
                windows,
                random=False, 
                window_shape=(1, 1, 1)):
        self.window_shape = window_shape
        self.windows = windows
        assert self.window_shape is not None, "window_shape cannot be None"
        assert len(self.window_shape) == 3, "window_shape must contain 3 elements"
        assert 1 <= window_shape[-1] <= ground_input.shape[-1], \
            "last dimension of window_shape must be in range 0..n_input_channels"
        assert window_shape[-1] == ground_input.shape[-1] or window_shape[-1] == 1, \
            "last element of window_shape must be 1 or equal to the last dimension of the input"
        assert ground_input.shape[1] % self.window_shape[0] == 0 and ground_input.shape[2] % self.window_shape[1] == 0, \
            "input dimensions must be multiple of window_shape dimensions"
        super(ImagePlayerIterator, self).__init__(ground_input, random)

    def _input_shape_merged(self):
        shape = list(self.input_shape)
        if self.window_shape[-1] > 1:
            shape[-1] = 1
        return shape

    def _assert_input_compatibility(self, ground_input):
        assert len(ground_input.shape) == 4, 'ImagePlayerIterator requires an input with 4 dimensions'

    def _get_number_of_players_from_shape(self):
        shape = self._input_shape_merged()
        if self.window_shape[0] > 1:
            shape[0] = shape[0] / self.window_shape[0]
        if self.window_shape[1] > 1:
            shape[1] = shape[1] / self.window_shape[1]
        print ('nplayers', np.prod(shape, dtype='int32'))
        return np.prod(shape, dtype='int32')
    
    def _get_masks_for_index(self, i):
        mask_input = np.zeros(self.input_shape, dtype=np.float32)
        mask = np.zeros(self.input_shape, dtype=np.float32)
        i = self.permutation[i]

        if self.windows:
            nrows, ncols = self.input_shape[0] // self.window_shape[0], self.input_shape[1] // self.window_shape[1]
            row_step = self.window_shape[0]
            col_step = self.window_shape[1]
            coalition_size = row_step*col_step
            row = i // nrows
            col = i % ncols

            mask_input[row*row_step:(1+row)*row_step, col*col_step:(1+col)*col_step] = 1
            mask[row*row_step:(1+row)*row_step, col*col_step:(1+col)*col_step] = 1. / coalition_size

            if self.window_shape[-1] > 1:
                mask_input = np.repeat(mask_input, self.input_shape[-1], -1)


        height, width = self.input_shape[1:]
        row = i // height
        col = i % width

        mask_input[:, row:(1 + row), col:(1 + col)] = 1
        mask[:, row:(1 + row), col:(1 + col)] = 1.0 / self.coalition_size

        return mask_input[np.newaxis], mask[np.newaxis]

    def get_explanation_shape(self):
        return self._input_shape_merged()



class WideDeepPlayerIterator(AbstractPlayerIterator):

    def __init__(self, 
                 ground_input, 
                 windows,
                 random: bool = False, 
                 window_shape = (1, 1, 1)) -> None:
        self.window_shape = window_shape
        super(WideDeepPlayerIterator, self).__init__(ground_input, windows, random)
        self.windows = windows

    def _input_shape_merged(self):
        shape = list(self.input_shape)
        if self.window_shape[-1] > 1:
            shape[-1] = 1
        return shape

    def _assert_input_compatibility(self, ground_input: Tuple[Tensor, Tensor]):
        ground_input, feats = ground_input
        assert (
            ground_input.dim() == 4
        ), "WidePNPlayerIterator requires an point cloud input with 3 dimensions."
        assert (
            feats.dim() == 2
        ), "WidePNPlayerIterator requires feature input with 2 dimensions."

    def _get_number_of_players_from_shape(self) -> int:

        shape = self._input_shape_merged()
        if self.window_shape[0] > 1:
            shape[0] = shape[0] / self.window_shape[0]
        if self.window_shape[1] > 1:
            shape[1] = shape[1] / self.window_shape[1]
        print ('nplayers', np.prod(shape, dtype='int32'))
        
        num_players_image = np.prod(self.input_shape, dtype='int32')
        num_players_feat = int(self.feat_shape[0])
        return num_players_image + num_players_feat

    def _get_masks_for_index(self, i:int):
        
        mask_input = np.zeros(self.input_shape, dtype=np.float32)
        mask = np.zeros(self.input_shape, dtype=np.float32)
        i = self.permutation[i]

        if self.windows: 
            nrows, ncols = self.input_shape[0] // self.window_shape[0], self.input_shape[1] // self.window_shape[1]
            row_step = self.window_shape[0]
            col_step = self.window_shape[1]
            coalition_size = row_step*col_step
            row = i // nrows
            col = i % ncols

            mask_input[row*row_step:(1+row)*row_step, col*col_step:(1+col)*col_step] = 1
            mask[row*row_step:(1+row)*row_step, col*col_step:(1+col)*col_step] = 1. / coalition_size

            if self.window_shape[-1] > 1:
                mask_input = np.repeat(mask_input, self.input_shape[-1], -1)
                
        else:
            height, width = self.input_shape[1:]
            row = i // height
            col = i % width

            mask_input[:, row:(1 + row), col:(1 + col)] = 1
            mask[:, row:(1 + row), col:(1 + col)] = 1.0 / self.coalition_size

        mask_feat = np.zeros(self.feat_shape, dtype=np.float32)
        mask_feat_out = np.zeros(self.feat_shape, dtype=np.float32)

        #mask_feat_out[i] = 1.0
        mask_feat_out[0] = 1.0
        mask_feat_out = mask_feat_out[np.newaxis]

        return ((mask_input[np.newaxis], mask_feat), (mask[np.newaxis], mask_feat_out))

    def _get_sampling_mask_for_indices(self, i:int):

        mask_input = np.zeros(self.input_shape, dtype=np.float32)
        mask = np.zeros(self.input_shape, dtype=np.float32)
        i = self.permutation[i]

        height, width = self.input_shape[1:]
        row = i // height
        col = i % width

        mask_input[:, row:(1 + row), col:(1 + col)] = 1
        mask[:, row:(1 + row), col:(1 + col)] = 1.0 / self.coalition_size

        mask_feat = np.zeros(self.feat_shape, dtype=np.float32)
        mask_feat_out = np.zeros(self.feat_shape, dtype=np.float32)

        #mask_feat_out[i] = 1.0
        mask_feat_out[0] = 1.0
        mask_feat_out = mask_feat_out[np.newaxis]

        return ((mask_input[np.newaxis], mask_feat), (mask[np.newaxis], mask_feat_out))


        


        

