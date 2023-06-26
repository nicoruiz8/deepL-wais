## Function to normalize or standarize (z-normalization) the provided 2-dimensional data matrix.
## Author: Nicolás Ruiz Lafuente

## The accepted types of normalization are:
# Min-max normalization: set each variable to range between -1 and 1.
# Standarization: set each variable to mean 0 and variance 1.

using Statistics

function data_normalization(X2d::Matrix; type::String)
    if type == "minmax"
        # Min-max normalization
        maxfactor = maximum(X2d, dims = 2)
        minfactor = minimum(X2d, dims = 2)
        X2dnorm = 2 .* ((X2d .- minfactor)./(maxfactor - minfactor)) .- 1
        return X2dnorm, maxfactor, minfactor

    elseif type == "std" 
        # Standarization (Z-normalization)
        meanfactor = mean(X2d, dims = 2)
        stdfactor = std(X2d, dims = 2)
        X2dnorm = (X2d .- meanfactor) ./ stdfactor
        return X2dnorm, meanfactor, stdfactor
    end
end
