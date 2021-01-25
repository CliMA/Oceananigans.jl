using Plots
using LaTeXStrings
using Printf
using Polynomials
using LinearAlgebra
using OffsetArrays

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.Models: ShallowWaterModel

include("RatesOfConvergence.jl")

using .RatesOfConvergence: rate_of_convergence, labels, shapes, colors, halos, plot_solutions!

### Model parameters and function

           U  = 1
           L  = 2.5
           W  = 0.1
           Ns = 2 .^ (6:10)
           Δt = 0.01 * minimum(L/Ns) / U

c(x, y, z, t, U, W) = exp( - (x - U * t)^2 / W^2 );

### Advection schemes

schemes = (
    CenteredSecondOrder(), 
    UpwindBiasedThirdOrder(), 
    CenteredFourthOrder(), 
    UpwindBiasedFifthOrder(), 
);

### Dictionaries to store errors and computed Rates of Convergence
error2 = Dict()
ROC2   = Dict()

pnorm = 1

for N in Ns, scheme in schemes

    grid = RegularCartesianGrid(Float64; size=(N, 1, 1), x=(-1, -1+L), y=(0, 1), z=(0, 1), halo=(halos(scheme), 1, 1))

    model = ShallowWaterModel(architecture = CPU(),
                                        grid = grid,
                                   advection = scheme,
                                    coriolis = nothing,
                  gravitational_acceleration = 0)
    
    set!(model, h = (x,y,z) -> c(x, y, z, 0, U, W) )
    
    simulation = Simulation(model, Δt=Δt, stop_iteration=1, iteration_interval=1)
    
    run!(simulation)

    c₁  = c.(grid.xC[:,1,1], grid.yC[1,1,1], grid.zC[1,1,1],  Δt, U, W);

    error2[(N, scheme)] = norm(abs.(model.solution.h[1:N,1,1] .- c₁[1:N]), pnorm)/N^(1/pnorm)
    
end

println(" ")        
println("Results are for the L"*string(pnorm)*"-norm:")
println(" ")        

for scheme in schemes
    
    name = labels(scheme)
    roc = rate_of_convergence(scheme)
    j = 3
    
    local best_fit2 = fit(log10.(Ns[2:end]),
                          log10.([error2[(N, scheme)] for N in Ns][2:end]), 1)

    ROC2[scheme] = best_fit2[1]
    println("Method = ", scheme, ", Rate of Convergence = ", @sprintf("%.2f", -ROC2[scheme]), ", Expected = ", roc)
    
end

plt2 = plot_solutions!(error2,
                       Ns,
                       schemes,
                       rate_of_convergence,
                       shapes,
                       colors,
                       labels,
                       pnorm,
                       ROC2)
savefig(plt2, "convergence_rates")

