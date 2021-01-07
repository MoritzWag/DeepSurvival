import numpy as np 
import pdb
import torch

from tqdm import tqdm

from src.dsap.coalition_policies.playergenerators import DefaultPlayerIterator
from src.dsap.layers.convolution import ProbConv2dInput
from src.dsap.layers.linear import ProbLinearInput
from lpdn import convert_to_lpdn, convert_layer

class DSAP():
    """
    """
    
    def __init__(self,
                 player_generator,
                 input_shape,
                 lpdn_model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.player_generator = player_generator
        self.input_shape = input_shape
        self.lpdn_model = lpdn_model.to(self.device).float()

    def run(self, images, tabular_data, baselines, steps=None):
        """
        """
        images = images.cpu()
        baselines = baselines.cpu()
        player_generator = self.player_generator
        if player_generator is None:
            player_generator = DefaultPlayerIterator(images)
        player_generator.set_n_steps(steps)
        ks = player_generator.get_steps_list()
        n_steps = len(ks)

        result = None
        batch_size = images.size()[0]
        batch_size_feat = tabular_data.size()[0]

        tile_input = [n_steps] + (len(images.shape) - 1) * [1]
        tile_mask = [n_steps * batch_size] + (len(images.shape) - 1) * [1]

        # enable heterogeneous input:
        if tabular_data is not None:
            tile_structured_mask = [n_steps * batch_size_feat] + (len(tabular_data.shape) - 1) * [1]
            tile_structured_input = [n_steps] + (len(tabular_data.shape) - 1) * [1]
            ks = [x - 2 for x in ks]
            ks_feat = [2] * len(ks)
            # ks_feat = ks * tabular_data.size()[1] / self.player_generator.n_players
            # pdb.set_trace()
            # ks = ks * images.size()[2] / self.player_generator.n_players
        
        ks = torch.as_tensor(ks, device=images.device)
        ks_feat = torch.as_tensor(ks_feat, device=images.device)

        self.lpdn_model.eval()

        with torch.no_grad():
            with tqdm(range(self.player_generator.n_players)) as progress_bar:
                for i, (mask, mask_output) in enumerate(self.player_generator):
                    # Workaround: as Keras requires the first dimension of the inputs to be the same,
                    # we tile and repeat the input, mask and ks vector to have them aligned.
                    if tabular_data is not None:
                        mask, feat_mask = mask
                        mask_output, feat_output = mask_output
                        #mask_output = np.stack((mask_output, feat_output), axis=1)
                    
                    mask = torch.as_tensor(mask, device=images.device)
                    feat_mask = torch.as_tensor(feat_mask, device=images.device)
                    
                    # here, it must be defined how the baseline is determined
                    inputs = (
                        torch.tensor(np.tile(images, tile_input)).to(self.device),
                        torch.tensor(np.tile(mask, tile_mask)).to(self.device),
                        torch.tensor(np.repeat(ks, images.shape[0])[:, np.newaxis]).to(self.device),
                    )
                    baseline_images = torch.tensor(np.tile(baselines, tile_input)).to(self.device)
                    
                    if tabular_data is not None:
                        structured_inputs = (
                            tabular_data.repeat(tile_structured_input).to(self.device).float(),
                            feat_mask.repeat(tile_structured_mask).to(self.device).float(),
                            ks_feat.repeat(batch_size_feat)
                            .unsqueeze(dim=1)
                            .to(self.device).float(),
                        )

                        out1, out2 = self.lpdn_model(unstructured=inputs, 
                                                     structured=structured_inputs,
                                                     baselines=baseline_images).to(self.device)
                    else:
                        out1, out2 = self.lpdn_model(unstructured=inputs,
                                                     baselines=baseline_images).to(self.device)

                    y1 = out1[0].cpu().detach().numpy()
                    y2 = out2[0].cpu().detach().numpy()
                    y1 = y1.reshape(n_steps, images.shape[0], -1)
                    y2 = y2.reshape(n_steps, images.shape[0], -1)
                    y = np.mean(y2 - y1, axis=0)

                    if np.isnan(y).any():
                        raise RuntimeError('Result contains nans! This should not happen...')
                    # Compute Shapley Values as mean of all coalition sizes
                    if result is None:
                        result = np.zeros(y.shape + mask_output.shape)

                    shape_mask = [1] * len(y.shape)
                    shape_mask += list(mask_output.shape)

                    shape_out = list(y.shape)
                    shape_out += [1] * len(mask_output.shape)

                    result += np.reshape(y, shape_out) * mask_output
        result = result.reshape(images.shape)
        return result


    def plot_attribution_maps(self):
        pass