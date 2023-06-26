## Source code for the work in "Machine learning for weather and climate prediction" in the 
## context of Trabajo de Fin de Grado (TFG) of the DEGREE IN PHYSICS at Universidad Complutense de Madrid (UCM). 

## Author: Nicolás Ruiz Lafuente
## Special thanks to Jan-Christophe Swierczek-Jereczek for his invaluable help and expertise in the matter.

## This script tests the custom neural network built for the work. For building and training processes check the README file.

using Flux, CairoMakie, BSON


## ----------------------------------- Load neural network -----------------------------------
BSON.@load "myNN.bson" model # optimised model
BSON.@load "testdata.bson" Xtest Ytest # test data already split
BSON.@load "lossvsepoch.bson" train_loss val_loss # evolution of loss vs epoch from training loop


## ------------------------- Creation and loading of needed funtions -------------------------
# Include F-score test measure
include("fscore.jl")

# Same loss function as the one used for the training
loss(m, x::Matrix, y::Matrix) = Flux.Losses.binarycrossentropy( m(x), y )

# Function to print results
function print_results(x, y)
    println("----------------------------------")
    println("   Comparing ̂y and y for some TEST data points yields:")
    yhat = model(x)
    n_label_dev = size(x,2)
    for j in vcat(1:10, n_label_dev-10:n_label_dev)
        println("   j = $j: m(x) = $(yhat[:, j][1]),  y = $(y[1, j])")
    end
    println("----------------------------------")
    println("   Loss is $(loss(model, x, y))")
    println("----------------------------------")
end


## -------------------------------------- Testing ----------------------------------------
# Network performance for the test data
print_results(Xtest, Ytest)
fscore("test", model, Xtest, Ytest; ϵ = 0.1f0)


## ----------------------------------- Loss vs epoch -------------------------------------
# Plot the evolution of training and validation losses through epochs
fig = Figure();
ax1 = Axis(fig[1,1], yscale = log10, xlabel = "Epoch", ylabel = "Loss");
lines!(ax1, 0:n_epoch, train_loss[0:n_epoch], label = "Train loss", color = RGBf(0, 0.6, 0.87));
lines!(ax1, 0:n_epoch, val_loss[0:n_epoch], label = "Validation loss", color = RGBf(1, 0.12, 0.36));
axislegend();
fig
