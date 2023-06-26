## Function that reads a .jld2 file containing Yelmo data for our neural network.
## Yelmo data is decribed in the README file.
## Author: Nicolás Ruiz Lafuente

using JLD2

function loaddata(filename)
    data = JLD2.jldopen(filename)

    t = data["t"]
    dt = data["dt"]
    vars = data["vars"]
    X = data["X"]
    Ypulse = data["Ypulse"]
    Ystep = data["Ystep"]

    vars = Dict(vars[1] => 1, vars[2] => 2, vars[3] => 3, vars[4] => 4, vars[5] => 5, vars[6] => 6, vars[7] => 7)

    display(vars)
    return t, X, Ystep, vars
end
