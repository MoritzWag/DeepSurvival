import pdb
import sksurv
import numpy as np
import pandas as pd 
import os
import argparse
import mlflow
import torch
from torch import nn
from torch.utils import data

from typing import Any, List, Optional

from src.data.adni import ADNI
#from src.data.linear import LinearData
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.datasets import load_whas500
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
#from src.models.baseline import Linear
from torch.utils.data.dataloader import default_collate
#from src.data.utils import cox_collate_fn



def parse_args():
    parser = argparse.ArgumentParser(description="generic runner for sksurv CoxPH model")

    parser.add_argument('--download', type=bool, default=False, metavar='N',
                        help='if trained in conjunction with DeepSurvival then download = False')
    parser.add_argument('--seed', type=int, default=1328, metavar='N',
                        help="seed for training")
    parser.add_argument('--experiment_name', type=str, default="sksurv", metavar='N')
    parser.add_argument('--run_name', type=str, default="sksurv", metavar='N')
    parser.add_argument('--split', type=int, default=0, help="which cv split to take")
    
    args = parser.parse_args()
    return args

def main(args):


    class LinearData(data.Dataset):
        """
        """
        features_list = ['ABETA', 'APOE4', 'AV45',
                        'C(PTGENDER)[T.Male]',
                        'FDG', 'PTAU', 
                        'TAU', 'real_age', 'age_male',
                        'bs_1', 'bs_2', 'bs_3', 'bs_4',
                        '.Linear', '.Quadratic', '.Cubic',
                        'C(ABETA_MISSING)[T.1]',
                        'C(TAU_MISSING)[T.1]',
                        'C(PTAU_MISSING)[T.1]',
                        'C(FDG_MISSING)[T.1]',
                        'C(AV45_MISSING)[T.1]'
                        ]
        
        def __init__(self, root,
                    part='train',
                    base_folder='adni',
                    split=0,
                    seed=1328):
        
            self.root = root
            self.part = part 
            self.split = split
            self.base_folder = base_folder
            self.seed = seed
            self.final_path = os.path.join(self.root, self.base_folder)

            self.df = self.load_dataset(path=self.final_path, part=self.part, split=self.split)
            self.eval_data = self.prepare_for_eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def __len__(self):
            return self.df.shape[0]
        
        def __getitem__(self, index):
            event = torch.tensor(self.df['event'].to_numpy()[:, np.newaxis][index]).to(self.device)
            time = torch.tensor(self.df['time'].to_numpy()[index]).to(self.device)
            tabular_data = torch.tensor(self.df[self.features_list].to_numpy()[index, :]).to(self.device)
            
            return tabular_data, event, time

        def load_dataset(self, path, part, split):
            """
            """
            if part != 'test':
                df = pd.read_csv(f"{path}/df_{part}_{split}.csv")
            else:
                df = pd.read_csv(f"{path}/df_{part}.csv")

            return df 

        def prepare_for_eval(self):
            """
            """
            y = []
            times = self.df['time']
            times_unique = np.unique(self.df['time'])
            times_unique[-1] -= 0.01
            events = self.df['event']
            for time, status in zip(times, events):
                instance = (bool(status), time)
                y.append(instance)

            dt = np.dtype('bool, float')
            y = np.array(y, dtype=dt)

            return {'y': y, 'times_unique': times_unique}





    # start mlflow run
    mlflow.set_experiment(args.experiment_name)

    features = ['ABETA', 'APOE4', 'AV45',
                'C(PTGENDER)[T.Male]',
                'FDG', 'PTAU', 
                'TAU', 
                'real_age', 
                'age_male',
                'bs_1', 'bs_2', 'bs_3', 'bs_4',
                '.Linear', '.Quadratic', '.Cubic', 
                'C(ABETA_MISSING)[T.1]',
                'C(TAU_MISSING)[T.1]',
                'C(PTAU_MISSING)[T.1]',
                'C(FDG_MISSING)[T.1]',
                'C(AV45_MISSING)[T.1]'
                ]

    # train_data = ADNI(root='./data',
    #                     part='train',
    #                     transform=False,
    #                     download=True,
    #                     base_folder='adni2d',
    #                     data_type='coxph',
    #                     simulate=False,
    #                     split=args.split,
    #                     seed=args.seed)

    train_data = LinearData(root="./data",
                            part='train',
                            base_folder='adni2d',
                            split=0)

    X_train = train_data.df
    X_train = X_train[features].to_numpy()
    y_train = train_data.eval_data['y']

    # val_data = ADNI(root='./data',
    #                 part='val',
    #                 transform=False,
    #                 download=args.download,
    #                 base_folder='adni2d',
    #                 data_type='coxph',
    #                 simulate=False,
    #                 split=args.split,
    #                 seed=args.seed)

    val_data = LinearData(root="./data",
                            part='val',
                            base_folder='adni2d',
                            split=0)

    X_val = val_data.df
    X_val = X_val[features].to_numpy()

    y_val = val_data.eval_data['y']

    # test_data = ADNI(root='./data',
    #                 part='test',
    #                 transform=False,
    #                 download=args.download,
    #                 base_folder='adni2d',
    #                 data_type='coxph',
    #                 simulate=False,
    #                 split=args.split,
    #                 seed=args.seed)

    test_data = LinearData(root="./data",
                            part='test',
                            base_folder='adni2d',
                            split=0)

    df_test = test_data.df
    X_test = df_test[features].to_numpy()
    y_test = test_data.eval_data['y']


    # fit the model
    estimator = CoxPHSurvivalAnalysis(alpha=0.00001).fit(X_train, y_train)


    # make predictions
    preds = estimator.predict(X_test)

    # evaluate 
    cindex = concordance_index_censored(df_test['event'].astype(np.bool), df_test['time'], preds)

    print(cindex)

    # retrieve coefficients
    coefficients = estimator.coef_

    # store coefficients for initalization
    storage_path = os.path.expanduser("linear_weights")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    np.save(file=f"{storage_path}/weights.npy", arr=coefficients)    

    mlflow.log_metric("cindex_tabular", cindex[0])
    mlflow.log_param('experiment_name', args.experiment_name)
    mlflow.log_param('run_name', args.run_name)
    

    def safe_normalize(x):
        """Normalize risk scores to avoid exp underflowing.

        Note that only risk scores relative to each other matter.
        If minimum risk score is negative, we shift scores so minimum
        is at zero.
        """
        x_min, _ = torch.min(x, dim=0)
        c = torch.zeros(x_min.shape, device=x.device)
        norm = torch.where(x_min < 0, -x_min, c)
        return x + norm

    def cox_collate_fn(batch, time_index=-1, data_collate=default_collate):
        """Create risk set from batch
        """
        transposed_data = list(zip(*batch))
        y_time = np.array(transposed_data[time_index])
        
        data = []
        for b in transposed_data:
            bt = data_collate(b)
            data.append(bt)
        
        data.append(torch.from_numpy(make_riskset(y_time)))

        return {'tabular_data': data[0], 
                'event': data[1], 'time': data[2], 'riskset': data[3]}

    def make_riskset(time):
        """Compute mask that represents each sample's risk set.
        Args:
            time: {np.ndarray} Observed event time sorted in descending order
        
        Returns:
            riskset {np.ndarray} Boolean matrix where the i-th row denotes the
            risk set of the i-th  instance, i.e. the indices j for which the observer time
            y_j >= y_i
        """

        assert time.ndim == 1
        #sort in descending order
        o = np.argsort(-time, kind="mergesort")
        n_samples = time.shape[0]
        risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
        for i_org, i_sort in enumerate(o):
            ti  = time[i_sort]
            k = i_org
            while k < n_samples and ti == time[o[k]]:
                k += 1
            risk_set[i_sort, o[:k]] = True

        return risk_set
        
    

    class LinearCox(nn.Module):
        """
        """
        def __init__(self,
                     structured_input_dim,
                     output_dim,
                     **kwargs):
            super(LinearCox, self).__init__()

            self.structured_input_dim = structured_input_dim
            self.output_dim = output_dim
            self.linear = nn.Linear(in_features=self.structured_input_dim, out_features=self.output_dim, bias=False)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        def forward(self, tabular_data, **kwargs):
            riskscore = self.linear(tabular_data.float())
            return riskscore

        def _loss_function(self, event, riskset, predictions):
            """
            """
            predictions = safe_normalize(predictions)
            pred_t = predictions.t()

            rr = self.logsumexp_masked(pred_t, riskset, axis=1, keepdim=True)

            #losses = torch.multiply(event, rr - predictions)
            losses = event * (rr - predictions)
            loss = torch.mean(losses)

            return loss
    
        def logsumexp_masked(self, risk_scores, mask, axis=0, keepdim=True):
            """
            """
            mask = mask.to(self.device)
            risk_scores_masked = risk_scores * mask
            amax = torch.max(risk_scores_masked, dim=axis, keepdim=True)
            risk_scores_shift = risk_scores_masked - amax[0]

            #exp_masked = torch.multiply(torch.exp(risk_scores_shift), mask)
            exp_masked = risk_scores_shift.exp() * mask
            exp_sum = torch.sum(exp_masked, axis=axis, keepdim=True)
            output = amax[0] + torch.log(exp_sum)
            if not keepdim:
                output = torch.squeeze(output, axis=axis)
            return output
    

    def logsumexp_masked(risk_scores, mask, axis=0, keepdim=True):
        """
        """
        mask = mask.to('cuda')
        risk_scores_masked = risk_scores * mask
        amax = torch.max(risk_scores_masked, dim=axis, keepdim=True)
        risk_scores_shift = risk_scores_masked - amax[0]

        #exp_masked = torch.multiply(torch.exp(risk_scores_shift), mask)
        exp_masked = risk_scores_shift.exp() * mask
        exp_sum = torch.sum(exp_masked, axis=axis, keepdim=True)
        output = amax[0] + torch.log(exp_sum)
        if not keepdim:
            output = torch.squeeze(output, axis=axis)
        return output
    


    class CoxphLoss(nn.Module):
        def forward(self, predictions: torch.Tensor, event: torch.Tensor, riskset: torch.Tensor) -> torch.Tensor:
            """Negative partial log-likelihood of Cox's proportional
            hazards model.

            Args:
                predictions (torch.Tensor):
                    The predicted outputs. Must be a rank 2 tensor.
                event (torch.Tensor):
                    Binary vector where 1 indicates an event 0 censoring.
                riskset (torch.Tensor):
                    Boolean matrix where the `i`-th row denotes the
                    risk set of the `i`-th instance, i.e. the indices `j`
                    for which the observer time `y_j >= y_i`.

            Returns:
                loss (torch.Tensor):
                    Scalar loss.

            References:
                .. [1] Faraggi, D., & Simon, R. (1995).
                A neural network model for survival data. Statistics in Medicine,
                14(1), 73–82. https://doi.org/10.1002/sim.4780140108
            """
            if predictions is None or predictions.dim() != 2:
                raise ValueError("predictions must be a 2D tensor.")
            if predictions.size()[1] != 1:
                raise ValueError("last dimension of predictions ({}) must be 1.".format(predictions.size()[1]))
            if event is None:
                raise ValueError("event must not be None.")
            if predictions.dim() != event.dim():
                raise ValueError(
                    "Rank of predictions ({}) must equal rank of event ({})".format(predictions.dim(), event.dim())
                )
            if event.size()[1] != 1:
                raise ValueError("last dimension event ({}) must be 1.".format(event.size()[1]))
            if riskset is None:
                raise ValueError("riskset must not be None.")

            event = event.type_as(predictions)
            riskset = riskset.type_as(predictions)
            predictions = safe_normalize(predictions)

            # move batch dimension to the end so predictions get broadcast
            # row-wise when multiplying by riskset
            pred_t = predictions.t()

            # compute log of sum over risk set for each row
            rr = logsumexp_masked(risk_scores=pred_t, mask=riskset, axis=1, keepdim=True)
            assert rr.size() == predictions.size()

            losses = event * (rr - predictions)
            loss = torch.mean(losses)

            return loss



    X, y = load_whas500()
    mask = X.notnull().all(axis=1)
    X, y = X.loc[mask], y[mask.values]

    Xe = OneHotEncoder().fit_transform(X)
    Xt = StandardScaler().fit_transform(Xe)

    class WhasDataset(data.Dataset):

        def __init__(self):
            X, y = load_whas500()
            mask = X.notnull().all(axis=1)
            X, y = X.loc[mask], y[mask.values]

            Xe = OneHotEncoder().fit_transform(X)
            Xt = StandardScaler().fit_transform(Xe).astype(np.float32)
            y_event = y["fstat"][:, np.newaxis].astype(np.uint8)
            y_time = y["lenfol"].astype(np.float32)

            self.n_features = Xt.shape[1]
            self.data = list(zip(Xt, y_event, y_time))

        def __getitem__(self, index: int):
            return self.data[index]

        def __len__(self) -> int:
            return len(self.data)


    torch.manual_seed(25)
    dev = torch.device('cuda')

    train_dataset = WhasDataset()
    train_loader = DataLoader(
            train_dataset, collate_fn=cox_collate_fn, batch_size=len(train_dataset)
    )

    model = LinearCox(structured_input_dim=train_dataset.n_features, output_dim=1).to(dev)
    opt = Adam(model.parameters(), lr=5e-4)
    loss_fn = CoxphLoss()

    model.train()
    for i in range(10000):
        for batch in train_loader:
            x = batch['tabular_data'].to(dev)
            y_event = batch['event'].to(dev)
            y_riskset = batch['riskset'].to(dev)

            opt.zero_grad()
            logits = model.forward(x)
            #loss = loss_fn(logits, y_event, y_riskset)
            loss = model._loss_function(y_event, y_riskset, predictions=logits)
            #loss = loss_fn(logits, y_event, y_riskset)
            loss.backward()
            opt.step()

        if i % 1000 == 0:
            print(i, loss.detach().cpu().numpy())

    # model.train()
    # for i in range(10000):
    #     for x, y_event, y_time, y_riskset in train_loader:
    #         pdb.set_trace()
    #         x = x.to(dev)
    #         y_event = y_event.to(dev)
    #         y_riskset = y_riskset.to(dev)

    #         opt.zero_grad()
    #         logits = model.forward(x)

    #         loss = loss_fn(logits, y_event, y_riskset)

    #         loss.backward()
    #         opt.step()

    #     if i % 1000 == 0:
    #         print(i, loss.detach().cpu().numpy())



    pdb.set_trace()





    # initialize model 
    model = LinearCox(structured_input_dim=21, output_dim=1).float().to('cuda')
    opt = Adam(model.parameters(), lr=5e-4)
    

    train_gen = DataLoader(dataset=train_data,
                            batch_size=len(train_data),
                            collate_fn=cox_collate_fn,
                            shuffle=True)
    
    test_gen = DataLoader(dataset=test_data,
                         batch_size=len(train_data),
                         collate_fn=cox_collate_fn,
                         shuffle=True
                         )

    loss_fn = CoxphLoss()

    model.train()
    for i in range(10000):
        for batch in train_gen:
            
            opt.zero_grad()
            #y_pred = model(**batch)
            y_pred = model(batch['tabular_data'])

            loss = loss_fn(y_pred, batch['event'], batch['riskset'])
            #loss = model._loss_function(batch['event'], batch['riskset'], predictions=y_pred)
            loss.backward()
            opt.step()
        
        if i % 100 == 0:
            print(i, loss)


    # make predictions
    model.eval()
    for batch in test_gen:
        preds = model(batch['tabular_data'])
        preds = preds.detach().cpu().numpy().squeeze()

        # evaluate 
        cindex_dl = concordance_index_censored(df_test['event'].astype(np.bool), df_test['time'], preds)
        print(cindex_dl)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)