const ReactantSimulation = Simulation{<:ReactantModel}

function Simulation(model::ReactantModel; Δt,
                    verbose = true,
                    stop_iteration = Inf)

   Δt = validate_Δt(Δt, architecture(model))

   diagnostics = OrderedDict{Symbol, AbstractDiagnostic}()
   output_writers = OrderedDict{Symbol, AbstractOutputWriter}()
   callbacks = OrderedDict{Symbol, Callback}()

   callbacks[:stop_iteration_exceeded] = Callback(stop_iteration_exceeded)

   # Convert numbers to floating point; otherwise preserve type (eg for DateTime types)
   #    TODO: implement TT = timetype(model) and FT = eltype(model)
   TT = eltype(model)
   Δt = Δt isa Number ? TT(Δt) : Δt

   stop_iteration = ConcreteRNumber(Float64(stop_iteration))

   return Simulation(model,
                     Δt,
                     stop_iteration,
                     nothing, # disallow stop_time
                     Inf,
                     diagnostics,
                     output_writers,
                     callbacks,
                     0.0,
                     false,
                     false,
                     verbose,
                     0.0)
end

function stop_iteration_exceeded(sim::ReactantSimulation)
    #=
    @trace if sim.model.clock.iteration >= sim.stop_iteration
        #=
        if sim.verbose
            msg = string("Model iteration ",
                         iteration(sim),
                         " equals or exceeds stop iteration ",
                         Int(sim.stop_iteration),
                         ".")

            @info wall_time_msg(sim) 
            @info msg
        end
        =#

        sim.running = false 
    end
    =#

    return nothing
end


