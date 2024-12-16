function coarse_grain(field, factor)
    N = size(field)[1]
    NN = N ÷ factor
    new_field = zeros(NN, NN)
    for i in 1:NN
        for j in 1:NN
            new_field[i, j] = mean(field[(i-1)*factor+1:i*factor, (j-1)*factor+1:j*factor])
        end
    end
    return new_field
end

function coarse_grained(field, factor)
    N = size(field)[1]
    new_field = zeros(N, N)
    NN = N ÷ factor
    for i in 1:NN 
        for j in 1:NN 
            is = (i-1)*factor+1:i*factor
            js = (j-1)*factor+1:j*factor
            new_field[is, js] .= mean(field[is, js])
        end
    end
    return new_field
end