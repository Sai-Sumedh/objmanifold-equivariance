from functions_geometry.manifold_analysis_correlation import *
from mftma.utils.make_manifold_data import make_manifold_data
from mftma.utils.activation_extractor import extractor
import torch

#Specify device 
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def get_dim_after_conv(prev_dim, kernel_size, stride):
        return int((prev_dim-kernel_size)/stride + 1)

def compute_manifold_properties(model, train_data, sampled_classes=10, examples_per_class=50, \
                                layer_types=['Conv2d', 'MaxPool2d', 'Linear'], dim_layer_activations=500):
    '''
    Given a model and train data, returns the manifold dimension, radius and 
        classification capacity of each class manifold
    Parameters:
    model- a nn.Module
    train_data- torch Dataset
    sampled_classes- number of class manifolds to analyse
    examples_per_class- number of points to use per manifold
    layer_types- types of layers in the model to analyse
    dim_layer_activations- dimension to which each layer's output is projected 
        (with a random projection) for computing manifold radius, dimension and capacity 

    Returns:
    a 5-tuple consisting of:
    capacities_allclasses: a list (one entry per model layer) of arrays 
        (with capacities of each class manifold)
    radii_allclasses: a list (one entry per model layer) of arrays 
        (with radii of each class manifold)
    dimensions_allclasses: a list (one entry per model layer) of arrays 
        (with dimensions of each class manifold) 
    correlations: a list with inter-manifold correlations for each layer
    layernames_all: a list of names of all layers for which manifold analysis has been done
    '''
    with torch.no_grad():
        data = make_manifold_data(train_data, sampled_classes, examples_per_class, seed=0)
        data = [d.to(device) for d in data]
        activations = extractor(model, data, layer_types=layer_types)
        
        np.random.seed(0)
        for layer, datax, in activations.items():
            X = [d.reshape(d.shape[0], -1).T for d in datax]
            # Get the number of features in the flattened data
            N = X[0].shape[0]
            # If N is greater than 500, do a random projection to 500 features
            if N > dim_layer_activations:
                print("Projecting {}".format(layer))
                M = np.random.randn(dim_layer_activations, N)
                M /= np.sqrt(np.sum(M*M, axis=1, keepdims=True))
                X = [np.matmul(M, d) for d in X]
            activations[layer] = X
        correlations = []
        capacities_allclasses = []
        radii_allclasses = []
        dimensions_allclasses = []
        layernames_all = activations.keys()
        for k, X, in activations.items():
            # Analyze each layer's activations
            am, rm, dm, r0, Km = manifold_analysis_corr(X, 0, 300, n_reps=1)
            capacities_allclasses.append(am)
            radii_allclasses.append(rm)
            dimensions_allclasses.append(dm)
            # Compute the mean values
            a = 1/np.mean(1/am)
            # r = np.mean(rm)
            r = np.median(rm)
            d = np.median(dm)
            print("{} capacity: {:4f}, radius {:4f}, dimension {:4f}, correlation {:4f}".format(k, a, r, d, r0))
            
            correlations.append(r0)

    return capacities_allclasses,radii_allclasses, dimensions_allclasses, correlations, layernames_all