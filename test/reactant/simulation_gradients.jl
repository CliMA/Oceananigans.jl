include(joinpath(@__DIR__, "..", "setup", "reactant_test_utils.jl"))

using Enzyme
using CUDA
using Statistics: mean

#####
##### Reverse-mode gradients of a compiled run! under each checkpointing scheme
#####
#
# Without checkpointing and with `Periodic`, `stop_iteration` is an `Int`, a constant of the
# program; with `Binomial` it is a device number, so one compiled gradient takes any number of
# steps. The model is a tracer diffusing at rest: the final tracer is linear in the initial one
# and the loss is quadratic, so a central difference of the eager loss is exact up to roundoff.

function diffusing_model(arch, timestepper)
    grid = RectilinearGrid(arch; size=(4, 4, 4), halo=(3, 3, 3), x=(0, 1), y=(0, 1), z=(0, 1))
    return HydrostaticFreeSurfaceModel(grid; timestepper,
                                       tracers = :T,
                                       buoyancy = nothing,
                                       closure = ScalarDiffusivity(κ=0.5),
                                       free_surface = ExplicitFreeSurface(gravitational_acceleration=1))
end

initial_temperature(x, y, z) = sinpi(2x) * cospi(2y) + z

function loss(simulation, T_init)
    model = simulation.model
    set!(model, T=T_init)
    run!(simulation)
    return mean(interior(model.tracers.T) .^ 2)
end

function grad_loss(simulation, dsimulation, T_init, dT_init)
    parent(dT_init) .= 0
    _, loss_value = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                                    loss, Enzyme.Active,
                                    Enzyme.Duplicated(simulation, dsimulation),
                                    Enzyme.Duplicated(T_init, dT_init))
    return dT_init, loss_value
end

# The loss of a fresh eager run of `stop_iteration` steps from the initial tracer `T₀`. A fresh
# Adams-Bashforth run takes a forward-Euler first step, as every compiled call does. The eager
# run! checks its stop criteria only after a step, so it takes one step at `stop_iteration = 0`:
# zero steps is no run! at all.
function eager_loss(T₀, stop_iteration, timestepper, Δt)
    model = diffusing_model(CPU(), timestepper)
    set!(model, T=T₀)
    simulation = Simulation(model; Δt, stop_iteration, verbose=false)
    stop_iteration > 0 && run!(simulation)
    return mean(interior(model.tracers.T) .^ 2)
end

function central_difference(T₀, index, stop_iteration, timestepper, Δt; ε=1e-3)
    T₊ = copy(T₀)
    T₋ = copy(T₀)
    T₊[index...] += ε
    T₋[index...] -= ε
    J₊ = eager_loss(T₊, stop_iteration, timestepper, Δt)
    J₋ = eager_loss(T₋, stop_iteration, timestepper, Δt)
    return (J₊ - J₋) / 2ε
end

@testset "Reactant Simulation: reverse-mode gradient of a compiled run!" begin
    Δt = 1e-2
    compiled_steps = 4
    probes = ((1, 1, 1), (3, 2, 4))

    for timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)
        T₀ = let model = diffusing_model(CPU(), timestepper)
            set!(model, T=initial_temperature)
            Array(interior(model.tracers.T))
        end

        gradients = Dict{String, Array{Float64, 3}}()

        for checkpointing in (false, Reactant.Periodic(2), Reactant.Binomial(2))
            label = string(checkpointing)
            @info "Testing the reverse-mode gradient of a compiled run! ($timestepper, checkpointing = $label)..."

            @testset "$timestepper, checkpointing = $label" begin
                r_model = diffusing_model(ReactantState(), timestepper)
                automatic_differentiation = (; checkpointing)
                simulation = Simulation(r_model; Δt, stop_iteration=compiled_steps, verbose=false, automatic_differentiation)
                traced_count = checkpointing isa Reactant.Binomial
                @test simulation.stop_iteration isa (traced_count ? ConcreteRNumber{Int} : Int)

                T_init = CenterField(r_model.grid)
                dT_init = CenterField(r_model.grid)
                set!(T_init, T₀)

                compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
                    simulation, Enzyme.make_zero(simulation), T_init, dT_init)

                # An Int stop_iteration is a constant of the program: only a device number can change
                # between calls. 0 is the empty loop and 6 is more steps than the binomial budget.
                counts = traced_count ? (compiled_steps, 0, 3, 6) : (compiled_steps,)

                for n in counts
                    simulation.stop_iteration = n
                    dT, loss_value = compiled_grad(simulation, Enzyme.make_zero(simulation), T_init, dT_init)
                    gradient = Array(interior(dT))

                    @test Reactant.to_number(loss_value) ≈ eager_loss(T₀, n, timestepper, Δt)
                    @test !any(isnan, gradient)
                    @test maximum(abs, gradient) > 0
                    for index in probes
                        @test gradient[index...] ≈ central_difference(T₀, index, n, timestepper, Δt) rtol=1e-8
                    end

                    n == compiled_steps && (gradients[label] = gradient)
                end
            end
        end

        # The three schemes differentiate the same program: their gradients agree to roundoff
        @test length(gradients) == 3
        if haskey(gradients, string(false))
            reference = gradients[string(false)]
            for (label, gradient) in gradients
                @test gradient ≈ reference rtol=1e-10
            end
        end
    end
end
