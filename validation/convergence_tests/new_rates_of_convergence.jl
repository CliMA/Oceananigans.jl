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

include("advection_properties.jl")
include("plot_rates_of_convergence.jl")

   U  = 1
   L  = 2.5
   W  = 0.1
   Ns = 2 .^ (6:10)
   Δt = 0.01 * minimum(L/Ns) / U
pnorm = 1

c(x, y, z, t, U, W) = exp( - (x - U * t)^2 / W^2 );

schemes = (
    CenteredSecondOrder(), 
    UpwindBiasedThirdOrder(), 
    CenteredFourthOrder(), 
    UpwindBiasedFifthOrder(), 
    WENO5()
);

error = Dict()
ROC   = Dict()

for N in Ns, scheme in schemes

    grid = RegularCartesianGrid(
        Float64; 
        size=(N, 1, 1), 
        x=(-1, -1+L), y=(0, 1), z=(0, 1), 
        halo=(halos(scheme), 1, 1))

    model = ShallowWaterModel(
        architecture = CPU(),
        grid = grid,
        advection = scheme,
        coriolis = nothing,
        gravitational_acceleration = 0)
    
    set!(model, h = (x,y,z) -> c(x, y, z, 0, U, W) )
    
    simulation = Simulation(model, Δt=Δt, stop_iteration=1, iteration_interval=1)
    
    run!(simulation)

    c₁  = c.(grid.xC[:,1,1], grid.yC[1,1,1], grid.zC[1,1,1],  Δt, U, W);

    error[(N, scheme)] = norm(abs.(model.solution.h[1:N,1,1] .- c₁[1:N]), pnorm)/N^(1/pnorm)   

end

println("\nResults are for the L"*string(pnorm)*"-norm:\n")

for scheme in schemes
    
    local best_fit = fit(log10.(Ns[2:end]),
                          log10.([error[(N, scheme)] for N in Ns][2:end]), 1)

    ROC[scheme] = best_fit[1]
    println(
        "Method = ", scheme, 
        ", Rate of Convergence = ", @sprintf("%.2f", -ROC[scheme]), 
        ", Expected = ", rate_of_convergence(scheme))
    
end

plt = plot_solutions!(
    error,
    Ns,
    schemes,
    rate_of_convergence,
    shapes,
    colors,
    labels,
    pnorm,
    ROC)
savefig(plt, "convergence_rates")
