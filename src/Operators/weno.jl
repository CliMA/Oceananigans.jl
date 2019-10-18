####
#### WENO-5 face interpolation functions
####

p₀(i, f) =  1/3 * f[i-1] + 5/6 * f[i]   -  1/6 * f[i+1]
p₁(i, f) = -1/6 * f[i-2] + 5/6 * f[i-1] +  1/3 * f[i]
p₂(i, f) =  1/3 * f[i-3] + 7/6 * f[i-2] + 11/6 * f[i-1]

####
#### WENO-5 weight calculation
####

β₀(i, f) = 13/12 * (f[i-1] - 2f[i]   + f[i+1])^2 + 1/4 * (3f[i-1] - 4f[i] + f[i+1])^2
β₁(i, f) = 13/12 * (f[i-2] - 2f[i-1] +   f[i])^2 + 1/4 * (f[i-2]  - f[i])^2
β₂(i, f) = 13/12 * (f[i-3] - 2f[i-2] + f[i-1])^2 + 1/4 * (f[i-3]  - 4f[i-2] + 3f[i-1])^2

###
### WENO-5 (stencil size 3) optimal weights
###

const C3₀ = 3/10
const C3₁ = 3/5
const C3₂ = 1/10

####
#### WENO-5 raw weights
####

const ϵ = 1e-6
const n = 2

α₀(i, f) = C3₀ / (β₀(i, f) + ϵ)^n
α₁(i, f) = C3₁ / (β₁(i, f) + ϵ)^n
α₂(i, f) = C3₂ / (β₂(i, f) + ϵ)^n

####
#### WENO-5 normalized weights
####

function weno_weights(i, f)
    a₀ = α₀(i, f)
    a₁ = α₁(i, f)
    a₂ = α₂(i, f)
    a_sum = a₀ + a₁ + a₂
    w₀ = a₀ / a_sum
    w₁ = a₁ / a_sum
    w₂ = a₂ / a_sum
    return w₁, w₂, w₂
end

F(i, f) = w₀(i, f) * p₀(i, f) + w₁(i, f) * p₁(i, f) + w₂(i, f) * p₂(i, f)
