using Random, LinearAlgebra, Distributions
import Distributions: MvNormal
import LinearAlgebra: cholesky, Symmetric

⊗(a, b) = a * b'
Random.seed!(1234)

# Here we go through an example where the theory "works perfectly"

# Define a linear forward map
# input Space 
Mo = 2
# output space
M = 2
H = randn(M, Mo)
forward_map(x) = H * x

# Create artificial solution
x = randn(Mo) .- 30
y̅ = forward_map(x)

# Define an initial ensemble distribution
MoMo = randn(Mo, Mo)
Σ = MoMo' * MoMo + I
prior = MvNormal(Σ)
# this choice implies prior ∝ exp( - x' Σ⁻¹ x)

# Implicitly define the likelihood function via the covariance 
MM = randn(M, M)
Γ = (MM' * MM + I)
# implies likelihood  exp(- T * dot(y - H * x, Γ⁻¹ * (y - H * x)) )
# where T is the final time of the simulation
# Observe that
# likelihood = exp(- dot(y - H * x, Γ⁻¹ * (y - H * x)) ) for T = 1

# number of steps
N = 100
# timestep size (h = 1/N implies T=1 at the final time)
h = 1 / N
# number of ensemble members
J = 4000 * Mo
# Likelihood perturbation for algorithm, (constructed from before)
ξ = MvNormal(1 / h * Γ)

# Construct Posterior
HΓHΣ = Symmetric((H' * (cholesky(Γ) \ H) + cholesky(Σ) \ I))
cHΓHΣ = cholesky(HΓHΣ) # LU fails
rhs = (H' * (cholesky(Γ) \ y̅[:]))
posterior_μ = cHΓHΣ \ rhs
posterior_Σ = cHΓHΣ \ I
posterior = MvNormal(posterior_μ, posterior_Σ)

# Construct empirical prior
u = [rand(prior) for i = 1:J]
u₀ = copy(u)

timeseries = []
push!(timeseries, copy(u))

# Viewed as a minimization problem, we are minimizing
# dot(y - H * x, Γ⁻¹ * (y - H * x)) + dot(x, Σ⁻¹ x)
# whose solution is 
# (H' Γ⁻¹ H + Σ⁻¹)⁻¹ H' Γ⁻¹ y
# Alternatively we can view the posterior as  
# posterior ∝ exp(-dot(y - H * x, Γ⁻¹ * (y - H * x)) - x' Σ⁻¹ x)
# which implies μ = (H' Γ⁻¹ H + Σ⁻¹)⁻¹ H' Γ⁻¹ y
# and           σ = (H' Γ⁻¹ H + Σ⁻¹)⁻¹

# EKI Algorithm
for i = 1:N
    u̅ = mean(u)
    G = forward_map.(u) # error handling needs to go here
    G̅ = mean(G)

    # define covariances
    Cᵘᵖ = (u[1] - u̅) ⊗ (G[1] - G̅)
    Cᵖᵖ = (G[1] - G̅) ⊗ (G[1] - G̅)
    for j = 2:J
        Cᵘᵖ += (u[j] - u̅) ⊗ (G[j] - G̅)
        Cᵖᵖ += (G[j] - G̅) ⊗ (G[j] - G̅)
    end
    Cᵘᵖ *= 1 / (J - 1)
    Cᵖᵖ *= 1 / (J - 1)

    # ensemblize the data
    y = [y̅ + rand(ξ) for i = 1:J]
    r = y - G

    # update
    Cᵖᵖ_factorized = cholesky(Symmetric(Cᵖᵖ + 1 / h * Γ))
    for j = 1:J
        u[j] += Cᵘᵖ * (Cᵖᵖ_factorized \ r[j])
    end
    push!(timeseries, copy(u))
end
