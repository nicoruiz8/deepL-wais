## Function that computes the accuracy (F-score) of a deep neural network for a classification problem.
## Author: Nicolás Ruiz Lafuente

function fscore(what::String, m, x, y; ϵ = 1.0f-3)
    yhat = m(x)
    tp = 0; fp = 0; tn = 0; fn = 0;
    for i in findall(>(0.5f0), y) # positive cases (1.0f0)
        if abs.(yhat[i]-y[i]) < ϵ
            tp += 1
        else
            fp += 1
        end
    end
    for i in findall(<(0.5f0), y) # negative cases (0.0f0)
        if abs.(yhat[i]-y[i]) < ϵ
            tn += 1
        else
            fn += 1
        end
    end
    fs = 2*tp / (2*tp+fp+fn)
    println("----------------------------------")
    println("Accuracy (F-score) with $what data is $(fs*100) %")
    println("----------------------------------")
    println("True positives: $tp")
    println("True negatives: $tn")
    println("False positives: $fp")
    println("False negatives: $fn")
    println("----------------------------------")
end