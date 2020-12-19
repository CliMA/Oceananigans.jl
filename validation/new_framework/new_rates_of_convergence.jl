using Plots
using LaTeXStrings
using Printf
using Polynomials
using LinearAlgebra

using Oceananigans.Grids
using Oceananigans.Advection

include("RatesOfConvergence.jl")
using .RatesOfConvergence

### Model parameters and function

           U  = 1
           L  = 2.5
           W  = 0.1
           Ns = 2 .^ (4:7)

           Δt = 0.01 * minimum(L/Ns) / U

c(x, t, U, W) = exp( -(x - U * t)^2 / W );

schemes = (
    UpwindBiasedFirstOrder, 
    CenteredSecondOrder, 
    UpwindBiasedThirdOrder, 
    CenteredFourthOrder, 
    UpwindBiasedFifthOrder, 
    CenteredSixthOrder
);

error  = Dict()
ROC    = Dict()

time_stepper = AdamsBashforth2
pnorm = 1

for N in Ns, scheme in schemes

    local grid = RegularCartesianGrid(size=(N, 1, 1), x=(-1, -1+L), y=(0, 1), z=(0, 1))
    xC = reshape(grid.xC, length(grid.xC), 1, 1)

    local c₋₁ = c.(xC, -Δt, U, W);
    local c₀  = c.(xC,   0, U, W);
    local c₁  = c.(xC,  Δt, U, W);

    local F₀ = zeros(N+2,1,1)
    local F₋₁= zeros(N+2,1,1)

    for i in 4:N-2, j in 1:grid.Ny, k in 1:grid.Nz
        
        #F₀[i, j, k] = advective_tracer_flux_x(i, j, k, grid, scheme(), U, c₀)
        #F₋₁[i, j, k] = advective_tracer_flux_x(i, j, k, grid, scheme(), U, c₋₁)
        F₀[i, j, k] = advective_flux(i, j, k, grid, scheme(), U, c₀)
        F₋₁[i, j, k] = advective_flux(i, j, k, grid, scheme(), U, c₋₁)
        
    end

    local cₛᵢₘ = zeros(N)
    local cₑᵣᵣ = zeros(N)

    for i in 4:N-1
        
        cₛᵢₘ[i] = one_time_step!(i, c₀, F₀, F₋₁, grid.Δx, Δt, time_stepper())
        cₑᵣᵣ[i] = cₛᵢₘ[i] - c₁[i]
        
    end
    
    error[(N, scheme)] = norm(cₑᵣᵣ, pnorm)/N^(1/pnorm)

end

### Compute rates of convergence

println(" ")        
println("Results are for the L"*string(pnorm)*"-norm:")
println(" ")        
for scheme in schemes
    
    name = labels(scheme())
    roc = rate_of_convergence(scheme())
    j = 3
    local best_fit = fit(log10.(Ns[2:end]), log10.([error[(N, scheme)] for N in Ns][2:end]), 1)
    ROC[scheme] = best_fit[1]
    println("Method = ", name, ", Rate of Convergence = ", @sprintf("%.2f", -ROC[scheme]), ", Expected = ", roc)
    
end

plot_solutions!(error, Ns, schemes, rate_of_convergence, shapes, colors, labels, pnorm, ROC)
