module Simulations

export Simulation, run!

using Oceananigans.Models
using Oceananigans.Utils

mutable struct Simulation{M, Δ, S, SI, ST, W, R, D, O, P, F}
                 model :: M
                    Δt :: Δ
         stop_criteria :: S
        stop_iteration :: SI
             stop_time :: ST
       wall_time_limit :: W
              run_time :: R
           diagnostics :: D
        output_writers :: O
              progress :: P
    progress_frequency :: F
end

function Simulation(model; Δt,
        stop_criteria = Function[iteration_limit_exceeded, stop_time_exceeded, wall_time_limit_exceeded],
       stop_iteration = Inf,
            stop_time = Inf,
      wall_time_limit = Inf,
          diagnostics = OrderedDict{Symbol, AbstractDiagnostic}(),
       output_writers = OrderedDict{Symbol, AbstractOutputWriter}(),
             progress = nothing,
   progress_frequency = 1
   )

   if stop_iteration == Inf && stop_time == Inf && wall_time_limit == Inf
         @warn "This simulation will run forever as stop iteration = stop time " *
               "= wall time limit = Inf."
   end

   run_time = 0.0

   return Simulation(model, Δt, stop_iteration, stop_time, wall_time_limit,
                     run_time, diagnostics, output_writers, progress)
end

function stop(sim)
    time_before = time()

    for sc in sim.stop_criteria
        if sc(sim)
            time_after = time()
            sim.run_time += time_after - time_before
            return true
        end
    end

    time_after = time()
    sim.run_time += time_after - time_before

    return false
end

function iteration_limit_exceeded(sim)
    if sim.model.clock.iteration > sim.stop_iteration
          @warn "Simulation is stopping. Model iteration $(sim.model.clock.iteration) " *
                "has exceeded simulation stop iteration $(sim.stop_iteration)."
          return true
    end
    return false
end

function stop_time_exceeded(sim)
    if sim.model.clock.time > sim.stop_time
          @warn "Simulation is stopping. Model time $(sim.model.clock.time) " *
                "has exceeded simulation stop time $(sim.stop_time)."
          return true
    end
    return false
end

function wall_time_limit_exceeded(sim)
    if sim.run_time > sim.wall_time_limit
          @warn "Simulation is stopping. Simulation run time $(sim.run_time) " *
                "has exceeded simulation wall time limit $(sim.wall_time_limit)."
          return true
    end
    return false
end

get_Δt(Δt) = Δt
get_Δt(wizard::TimeStepWizard) = wizard.Δt

function run!(sim)
    model = sim.model
    clock = model.clock

    while !stop(sim)
        time_before = time()

        if clock.iteration == 0
            [run_diagnostic(sim.model, diag) for diag in values(sim.diagnostics)]
            [write_output(sim.model, out)    for out  in values(sim.output_writers)]
        end

        for n in 1:sim.progress_frequency
            time_step!(model, Δt, euler=n==1)

            [time_to_run(clock, diag) && run_diagnostic(sim.model, diag) for diag in values(diagnostics)]
            [time_to_run(clock, out)  && write_output(sim.model, out)    for out  in values(output_writers)]
        end

        Δt isa TimeStepWizard && update_Δt!(Δt, model)
        progress isa Function && progress(simulation)

        time_after = time()
        sim.run_time += time_after - time_before
    end
end

end
