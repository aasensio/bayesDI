import numpy as np
import torch
import torch.nn.functional as F
from nflows import transforms, distributions, flows, utils
import nflows.nn.nets as nn_
import matplotlib.pyplot as pl
from modules import resnet

# https://github.com/stephengreen/lfi-gw/blob/master/lfigw/nde_flows.py

def create_linear_transform(input_dim):
    """Create the composite linear transform PLU.
    Arguments:
        input_dim {int} -- dimension of the space
    Returns:
        Transform -- nde.Transform object
    """
    
    permutation = transforms.RandomPermutation(features = input_dim)
    linear = transforms.LULinear(input_dim, identity_init=True)

    return transforms.CompositeTransform([permutation, linear])

def create_base_transform(i, 
                            input_dim, 
                            context_dim,
                            hidden_dim=512,
                            num_transform_blocks=2,
                            activation='relu',
                            dropout_probability=0.0,
                            batch_norm=False,
                            num_bins=8,
                            tail_bound=1.,
                            apply_unconditional_transform=False,
                            base_transform_type='rq-coupling',
                            transform_net='conv'):

    """Build a base NSF transform of x, conditioned on y.
    This uses the PiecewiseRationalQuadraticCoupling transform or
    the MaskedPiecewiseRationalQuadraticAutoregressiveTransform, as described
    in the Neural Spline Flow paper (https://arxiv.org/abs/1906.04032).
    Code is adapted from the uci.py example from
    https://github.com/bayesiains/nsf.
    A coupling flow fixes half the components of x, and applies a transform
    to the remaining components, conditioned on the fixed components. This is
    a restricted form of an autoregressive transform, with a single split into
    fixed/transformed components.
    The transform here is a neural spline flow, where the flow is parametrized
    by a residual neural network that depends on x_fixed and y. The residual
    network consists of a sequence of two-layer fully-connected blocks.
    Arguments:
        i {int} -- index of transform in sequence
        param_dim {int} -- dimensionality of x
    Keyword Arguments:
        context_dim {int} -- dimensionality of y (default: {None})
        hidden_dim {int} -- number of hidden units per layer (default: {512})
        num_transform_blocks {int} -- number of transform blocks comprising the
                                      transform (default: {2})
        activation {str} -- activation function (default: {'relu'})
        dropout_probability {float} -- probability of dropping out a unit
                                       (default: {0.0})
        batch_norm {bool} -- whether to use batch normalization
                             (default: {False})
        num_bins {int} -- number of bins for the spline (default: {8})
        tail_bound {[type]} -- [description] (default: {1.})
        apply_unconditional_transform {bool} -- whether to apply an
                                                unconditional transform to
                                                fixed components
                                                (default: {False})
        base_transform_type {str} -- type of base transform
                                     ([rq-coupling], rq-autoregressive)
    Returns:
        Transform -- the NSF transform
    """

    if activation == 'elu':
        activation_fn = F.elu
    elif activation == 'relu':
        activation_fn = F.relu
    elif activation == 'leaky_relu':
        activation_fn = F.leaky_relu
    else:
        activation_fn = F.relu   # Default
        print('Invalid activation function specified. Using ReLU.')

    if base_transform_type == 'rq-coupling':

        mask = utils.create_alternating_binary_mask(input_dim, even=(i % 2 == 0))

        if (transform_net == 'fc'):
            transform_net = lambda in_features, out_features: nn_.ResidualNet(
                                                in_features = in_features,
                                                out_features = out_features,
                                                hidden_features = hidden_dim,
                                                context_features = context_dim,
                                                num_blocks = num_transform_blocks,
                                                activation = activation_fn,
                                                dropout_probability = dropout_probability,
                                                use_batch_norm = batch_norm)

        if (transform_net == 'conv'):
            transform_net = lambda in_features, out_features: resnet.ConvResidualNet1d(
                                            in_channels = 1,
                                            out_channels = out_features // in_features,
                                            hidden_channels = hidden_dim,
                                            context_channels = context_dim,
                                            num_blocks = num_transform_blocks,
                                            activation = activation_fn,
                                            dropout_probability = dropout_probability,
                                            use_batch_norm = batch_norm)

        transform = transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask = mask,
                transform_net_create_fn = transform_net,
                num_bins = num_bins,
                tails = 'linear',
                tail_bound = tail_bound,
                apply_unconditional_transform = apply_unconditional_transform
            )

    elif base_transform_type == 'rq-autoregressive':
        transform = transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=input_dim,
            hidden_features=hidden_dim,
            context_features=context_dim,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=activation_fn,
            dropout_probability=dropout_probability,
            use_batch_norm=batch_norm
        )
    else:
        raise ValueError

    return transform

def create_transform(input_dim, context_dim, num_flow_steps, base_transform_kwargs):
    """Build a sequence of NSF transforms, which maps parameters x into the
    base distribution u (noise). Transforms are conditioned on strain data y.
    Note that the forward map is f^{-1}(x, y).
    Each step in the sequence consists of
        * A linear transform of x, which in particular permutes components
        * A NSF transform of x, conditioned on y.
    There is one final linear transform at the end.
    This function was adapted from the uci.py example in
    https://github.com/bayesiains/nsf
    Arguments:
        num_flow_steps {int} -- number of transforms in sequence
        param_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        base_transform_kwargs {dict} -- hyperparameters for NSF step
    Returns:
        Transform -- the constructed transform
    """

    transform = transforms.CompositeTransform([
        transforms.CompositeTransform([
            create_linear_transform(input_dim),
            create_base_transform(i, input_dim, context_dim=context_dim, **base_transform_kwargs)
        ]) for i in range(num_flow_steps)] + [create_linear_transform(input_dim)])

    return transform

def fun(input_dim):
    
    return fun

def create_nsf_model(input_dim, context_dim, num_flow_steps, base_transform_kwargs, learn_normal=False):

    """Build NSF (neural spline flow) model. This uses the nsf module
    available at https://github.com/bayesiains/nsf.
    This models the posterior distribution p(x|y).
    The model consists of
        * a base distribution (StandardNormal, dim(x))
        * a sequence of transforms, each conditioned on y
    Arguments:
        input_dim {int} -- dimensionality of x
        context_dim {int} -- dimensionality of y
        num_flow_steps {int} -- number of sequential transforms
        base_transform_kwargs {dict} -- hyperparameters for transform steps
    Returns:
        Flow -- the model
    """
    
    # Define a base distribution.
    if (learn_normal):
        base_distribution = distributions.DiagonalNormal(shape=(input_dim,))
    else:
        base_distribution = distributions.StandardNormal(shape=(input_dim,))
    # if (sigma_base != 1):
    #     def fun2(x):            
    #         n_batch, n = x.shape
    #         return torch.cat([torch.zeros((n_batch, input_dim), device=x.device), sigma_base * torch.ones((n_batch, input_dim), device=x.device)], dim=1)
    #     base_distribution = distributions.ConditionalDiagonalNormal(shape=(input_dim,), context_encoder=fun2)
        
    # Define the neural spline transform
    transform = create_transform(input_dim, context_dim, num_flow_steps, base_transform_kwargs)

    # Create the flow
    flow = flows.Flow(transform=transform, distribution=base_distribution)

    # Add the hyperparameters for reconstructing the model after loading
    flow.model_hyperparams = {
        'input_dim': input_dim,
        'num_flow_steps': num_flow_steps,
        'context_dim': context_dim,
        'base_transform_kwargs': base_transform_kwargs
    }
    
    return flow

def obtain_samples(flow, y, nsamples, device=None, batch_size=512):
    """Draw samples from the posterior.
    Arguments:
        flow {Flow} -- NSF model
        y {array} -- strain data
        nsamples {int} -- number of samples desired
    Keyword Arguments:
        device {torch.device} -- model device (CPU or GPU) (default: {None})
        batch_size {int} -- batch size for sampling (default: {512})
    Returns:
        Tensor -- samples
    """

    with torch.no_grad():
        flow.eval()

        y = torch.from_numpy(y).unsqueeze(0).to(device)

        num_batches = nsamples // batch_size
        num_leftover = nsamples % batch_size

        samples = [flow.sample(batch_size, y) for _ in range(num_batches)]
        if num_leftover > 0:
            samples.append(flow.sample(num_leftover, y))

        # The batching in the nsf package seems screwed up, so we had to do it
        # ourselves, as above. They are concatenating on the wrong axis.

        # samples = flow.sample(nsamples, context=y, batch_size=batch_size)

        return torch.cat(samples, dim=1)[0]
    

if (__name__ == '__main__'):
    
    base_transform_kwargs = {
                        'hidden_dim': 50,
                        'num_transform_blocks': 2,
                        'activation': 'relu',
                        'dropout_probability': 0.0,
                        'batch_norm': False,
                        'num_bins': 10,
                        'tail_bound': 3.0,
                        'apply_unconditional_transform': False
                    }
    model = create_nsf_model(20, 1, 3, base_transform_kwargs)

    # context = np.array([[2.]])
    # context = torch.tensor(context.astype('float32'))

    # samples = model.sample(5000, context).detach().cpu().numpy()
    # pl.plot(samples[0,:,0], samples[0,:,1], '.')
    # pl.show()