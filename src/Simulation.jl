module Simulation

export Simulation, run!

using Oceananigans.Models
using Oceananigans.Utils

mutable struct Simulation{M, T, SI, ST, W, R, D, O, P}
                 model :: M
                    Δt :: T
        stop_iteration :: SI
             stop_time :: ST
       wall_time_limit :: W
              run_time :: R
           diagnostics :: D
        output_writers :: O
              progress :: P
    progress_frequency :: F
end

function Simulation(model, Δt;
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

get_Δt(Δt) = Δt
get_Δt(wizard::TimeStepWizard) = wizard.Δt

function run!(sim)
      stop_simulation = false

      if sim.model.clock.iteration > sim.stop_iteration
            @warn "Simulation will not be run! Model iteration $(sim.model.clock.iteration) " *
                  "has exceeded simulation stop iteration $(sim.stop_iteration)."
            stop_simulation = true
      end

      if sim.model.clock.time > sim.stop_time
            @warn "Simulation will not be run! Model time $(sim.model.clock.time) " *
                  "has exceeded simulation stop time $(sim.stop_time)."
            stop_simulation = true
      end

      if sim.run_time > sim.wall_time_limit
            @warn "Simulation will not be run! Simulation run time $(sim.run_time) " *
                  "has exceeded simulation wall time limit $(sim.wall_time_limit)."
            stop_simulation = true
      end

      clock = sim.model.clock
      while !stop_simulation
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

            time_step!(sim.model, Δt=get_Δt(sim.Δt), Nt=sim.progress_frequency,
                       diagnostics=sim.diagnostics, output_writers=sim.output_writers)

            Δt isa TimeStepWizard && update_Δt!(Δt, model)
            progress isa Function && progress(simulation)

            time_after = time()
            sim.run_time += time_after - time_before

            sim.model.clock.iteration > sim.stop_iteration && stop_simulation = true
            sim.model.clock.time > sim.stop_time && stop_simulation = true
            sim.run_time > sim.wall_time_limit && stop_simulation = true
      end
end

end
