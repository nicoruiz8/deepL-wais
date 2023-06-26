## Source code for the work in "Machine learning for weather and climate prediction" in the 
## context of Trabajo de Fin de Grado (TFG) of the DEGREE IN PHYSICS at Universidad Complutense de Madrid (UCM). 

## Author: Nicolás Ruiz Lafuente
## Special thanks to Jan-Christophe Swierczek-Jereczek for his invaluable help and expertise in the matter.

## This script just builds and trains a custom neural network. For testing results check the README.

using Flux, CairoMakie, ProgressBars, Random, BSON


## ------------------------- Creation and loading of needed funtions -------------------------
# Function to read YELMO data
include("loaddata.jl")

# Function to normalize data
include("data_normalization.jl")

# Function to build the model
function build_NN(n_input::Int, p::Float32)
    return Chain(
        Dense(n_input => 11, leakyrelu, init=Flux.kaiming_uniform),
        Dropout(p),
        Dense(11 => 8, leakyrelu, init=Flux.kaiming_uniform),
        Dropout(p),
        Dense(8 => 5, leakyrelu, init=Flux.kaiming_uniform),
        Dropout(p),
        Dense(5 => 2, leakyrelu, init=Flux.kaiming_uniform),
        Dropout(p),
        Dense(2 => 1, sigmoid, init=Flux.glorot_uniform),
    )
end

# Loss function
loss(m, x::Matrix, y::Matrix) = Flux.Losses.binarycrossentropy( m(x), y )

# Function to print results
function print_results(when::String, what::String, x, y)
    println("----------------------------------")
    println("$when the training:")
    println("----------------------------------")
    println("   Comparing ̂y and y for some $what data points yields:")
    yhat = model(x)
    n_label_dev = size(x,2)
    for j in vcat(1:10, n_label_dev-10:n_label_dev)
        println("   j = $j: m(x) = $(yhat[:, j][1]),  y = $(y[1, j])")
    end
    println("----------------------------------")
    println("   Loss is $(loss(model, x, y))")
    println("----------------------------------")
end

# Function to split the data in train-validation-test
function data_split(x::Matrix, y::Matrix; at::Tuple)
    n_train = Int(round(at[1]*n_label))
    n_val = Int(round(at[2]*n_label))

    # Training data
    Xtrain = x[:, 1:n_train]
    Ytrain = y[:, 1:n_train]

    # Validation data
    Xval = x[:, n_train+1:n_train+n_val]
    Yval = y[:, n_train+1:n_train+n_val]

    # Test data
    Xtest = x[:, n_train+n_val+1:end]
    Ytest = y[:, n_train+n_val+1:end]

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest
end

# Function to test the accuracy
include("fscore.jl")


## ---------------------------------- Data Preparation ---------------------------------------
# Load the YELMO data
t, X, Ystep, vars = loaddata("neuralnet_yelmodata.jld2");

# Define dimensions: 
# n1 = number of variables and their time derivatives
# n2 = number of experiments
# n3 = time index
n1, n2, n3 = size(X)
n_label = n2*n3

# Reshape data so that it is 2-dimensional
Xperm = permutedims(X, (1, 3, 2))
Yperm = permutedims(Ystep, (1, 3, 2))
X2d = reshape(Xperm, n1, n_label)
Y2d = reshape(Yperm, 1, n_label)

# Normalization or standarization
X2dnorm, meanfactor, stdfactor = data_normalization(X2d; type = "minmax")

# Add random noise to the data
rng = MersenneTwister(1234);
noise = rand(rng, Float32, (n1,n_label)).*rand(rng,(-0.01f0,0.01f0), (n1,n_label));
X2dnorm += noise

# Data splitting
Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = data_split(X2dnorm, Y2d; at = (0.70, 0.15))

# Batches of training data
batchsize = 512
data_train = Flux.DataLoader((Xtrain, Ytrain), batchsize=batchsize, shuffle=true)


## -------------------------------------- Model building --------------------------------------
# Build the neural network
model = build_NN(n1, 0.1f0)
_, rebuild = Flux.destructure(model); # model's structure needed to rebuild at any time with custom parameters

# Network performance before training
print_results("Before", "train", Xtrain, Ytrain)


## ----------------------------------------- Training -----------------------------------------
# Number of epochs
n_epoch = 1000;

# Setup of the training optimiser
opt_alg = Flux.Optimise.RAdam(0.0001, (0.9, 0.999));
opt_state = Flux.setup(opt_alg, model);

# Data allocation
train_loss = fill(Inf32, 0:n_epoch);
train_loss[0] = loss(model, Xtrain, Ytrain);
val_loss = fill(Inf32, 0:n_epoch);
val_loss[0] = loss(model, Xval, Yval);
parameters = [];

# Training loop
for epoch in ProgressBar(1:n_epoch)
    # Train at each epoch
    Flux.train!(model, data_train, opt_state) do m, x, y
        loss(m, x, y)
    end

    # Save loss at each epoch
    train_loss[epoch] = loss(model, Xtrain, Ytrain)
    val_loss[epoch] = loss(model, Xval, Yval)
    
    # Save the parameters at each epoch
    p,_ = Flux.destructure(model)
    append!(parameters, [p])
end

# Keep the parameters for which validation loss is minimum
val_loss_min, idx_min = findmin(val_loss)

# Final optimized neural network
model = rebuild(parameters[idx_min])


## --------------------------------------- Performance ---------------------------------------
# Network performance after training for training data
print_results("After", "train", Xtrain, Ytrain)
fscore("train", model, Xtrain, Ytrain; ϵ = 0.1f0)

# Network performance after training for validation data
print_results("After", "validation", Xval, Yval)
fscore("validation", model, Xval, Yval; ϵ = 0.1f0)


## ----------------------------------------- Saving  -----------------------------------------
BSON.@save "myNN.bson" model
BSON.@save "testdata.bson" Xtest Ytest
BSON.@save "lossvsepoch.bson" train_loss val_loss