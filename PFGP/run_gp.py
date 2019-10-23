'''
When we use Jupyter, it's shared with others. It turns out that when we share,
we don't get enough memory. So we get an interactive GPU all to ourselves and
run it in python here.
'''

import gc
import pickle
import numpy as np
import torch
import gpytorch
from LBFGS import FullBatchLBFGS


# Verify the number of GPUs
n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))
output_device = torch.device('cuda:0')

# Pull training data
with open('../preprocessing/sdt/gasdb/feature_dimensions.pkl', 'rb') as file_handle:
    orig_atom_fea_len, nbr_fea_len = pickle.load(file_handle)
with open('../preprocessing/splits_gasdb.pkl', 'rb') as file_handle:
    splits = pickle.load(file_handle)
docs_train, docs_val = splits['docs_train'], splits['docs_val']
sdts_train, sdts_val = splits['sdts_train'], splits['sdts_val']
targets_train, targets_val = splits['targets_train'], splits['targets_val']
adsorbates = list({doc['adsorbate'] for doc in docs_val})
adsorbates.sort()

# Load CGCNN outputs
with open('penult.pkl', 'rb') as file_handle:
    cache = pickle.load(file_handle)
penult_train = cache['penult_train']
penult_val = cache['penult_val']
scaler = cache['scaler']
input_train = cache['input_train']
input_val = cache['input_val']
targets_train = cache['targets_train']
targets_val = cache['targets_val']

# Make the targets contiguous...?
targets_train, targets_val = targets_train.contiguous(), targets_val.contiguous()
# Move them to the GPU
input_train, targets_train = input_train.to(output_device), targets_train.to(output_device)
input_val, targets_val = input_val.to(output_device), targets_val.to(output_device)


# Define GP
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Define training routine
def train(train_x, train_y,
          n_devices, output_device,
          checkpoint_size, preconditioner_size,
          n_training_iter):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(train_x, train_y, likelihood, n_devices).to(output_device)
    model.train()
    likelihood.train()

    optimizer = FullBatchLBFGS(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
         gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()

        for i in range(n_training_iter):
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)

            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_training_iter, loss.item(),
                model.covar_module.module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))

            if fail:
                print('Convergence reached!')
                break

    print(f"Finished training on {train_x.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


# Define routine for getting GPU settings
def find_best_gpu_setting(train_x, train_y,
                          n_devices, output_device,
                          preconditioner_size):
    N = train_x.size(0)

    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            _, _ = train(train_x, train_y,
                         n_devices=n_devices, output_device=output_device,
                         checkpoint_size=checkpoint_size,
                         preconditioner_size=preconditioner_size, n_training_iter=1)

            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
    return checkpoint_size


# Set a large enough preconditioner size to reduce the number of CG iterations run
preconditioner_size = 100
checkpoint_size = find_best_gpu_setting(input_train, targets_train,
                                        n_devices=n_devices,
                                        output_device=output_device,
                                        preconditioner_size=preconditioner_size)

# Fit and save the model
model, likelihood = train(input_train, targets_train,
                          n_devices=n_devices, output_device=output_device,
                          checkpoint_size=10000,
                          preconditioner_size=100,
                          n_training_iter=20)
torch.save(model.state_dict(), 'model_state.pth')

# Make and save the predictions
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
    preds = model(input_val)
targets_pred = preds.mean
targets_std = preds.stddev.detach().cpu().numpy()
residuals = (targets_pred - targets_val).detach().cpu().numpy()
with open('predictions.pkl', 'wb') as file_handle:
    pickle.dump((residuals, targets_std), file_handle)
