using Plots
using LaTeXStrings
using Printf
using Polynomials
using LinearAlgebra
using OffsetArrays

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection

include("RatesOfConvergence.jl")

using .RatesOfConvergence: ForwardEuler, AdamsBashforth2
using .RatesOfConvergence: update_solution #one_time_step!

using .RatesOfConvergence: UpwindBiasedFirstOrder, CenteredSixthOrder
using .RatesOfConvergence: advective_flux, rate_of_convergence, labels, shapes, colors, halos

using .RatesOfConvergence: plot_solutions!

### Model parameters and function

           U  = 1
           L  = 2.5
           W  = 0.1
           Ns = 2 .^ (3:7)

           Δt = 0.01 * minimum(L/Ns) / U

c(x, y, z, t, U, W) = exp( -(x - U * t)^2 / W );

schemes = (
#    UpwindBiasedFirstOrder, 
    CenteredSecondOrder, 
#    UpwindBiasedThirdOrder, 
#    CenteredFourthOrder, 
#    UpwindBiasedFifthOrder, 
#    CenteredSixthOrder
);

error  = Dict()
ROC    = Dict()

error2 = Dict()

time_stepper = AdamsBashforth2
pnorm = 1

for N in Ns, scheme in schemes

    grid = RegularCartesianGrid(size=(N, 1, 1), x=(-1, -1+L), y=(0, 1), z=(0, 1), halo=(halos(scheme()), 1, 1))

    ### Using Oceananigans
    model = IncompressibleModel(architecture = CPU(),
                                 timestepper = :RungeKutta3,
                                        grid = grid,
                                   advection = scheme(),
                                    coriolis = nothing,
                                    buoyancy = nothing,
                                     tracers = :c,
                                     closure = nothing)


    set!(model, u = U,
         c = (x, y, z) -> c(x, y, z, 0, U, W) )

    simulation = Simulation(model, Δt=Δt, stop_iteration=1, iteration_interval=1)

    run!(simulation)

    ### Using explicit method
    cₛᵢₘ = update_solution(c, U, W, Δt, grid, scheme, time_stepper)
        
    c₁  = c.(grid.xC[:,1,1], grid.yC[1,1,1], grid.zC[1,1,1],  Δt, U, W);
    c1ₑᵣᵣ = zeros(N)
    c2ₑᵣᵣ = zeros(N)
    
    for i in 2:N-1
        
        c1ₑᵣᵣ[i] = cₛᵢₘ[i] - c₁[i]
        c2ₑᵣᵣ[i] = model.tracers.c[i] - c₁[i]
        
    end

    error[ (N, scheme)] = norm(c1ₑᵣᵣ, pnorm)/N^(1/pnorm)
    error2[(N, scheme)] = norm(c2ₑᵣᵣ, pnorm)/N^(1/pnorm)

    #=
    plt1 = plot(
        grid.xC,
        cₛᵢₘ,
        lw=3,
        linecolor=:blue,
        label="Explicit Method",
        title="Solutions at Δt"
    )
    plot!(
        plt1,
        grid.xC,
        model.tracers.c.data[:,1,1],
        lw=2,
        linecolor=:red,
        label="IncompressibleModel"
    )
    #display(plt1)
    savefig(plt1, "tmp1")
    
    #xc = xnodes(model.tracers.c)
    plt2 = plot(grid.xC[1:end-1],
                c1ₑᵣᵣ, 
                lw=3,
                linecolor=:blue,
                label="Explicit Method",
                title="Errors at Δt"
               )
    plot!(plt2, grid.xC[1:end-1],
               c2ₑᵣᵣ,
               lw=2,
               linecolor=:red,
               label="IncompressibleModel"
               )
    #display(plt2)
    savefig(plt2, "tmp2")
    =#
    
end

print("error\n")
print([error[(N, CenteredSecondOrder)] for N in Ns])

print("error2\n")
print([error2[(N, CenteredSecondOrder)] for N in Ns])

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
