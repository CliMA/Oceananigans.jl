const ReactantSimulation = Simulation{<:ReactantModel}

function Simulation(model::ReactantModel; Δt,
                    verbose = true,
                    stop_iteration = Inf,
                    wall_time_limit = Inf,
                    minimum_relative_step = 0)

   Δt = validate_Δt(Δt, architecture(model))

   diagnostics = OrderedDict{Symbol, AbstractDiagnostic}()
   output_writers = OrderedDict{Symbol, AbstractOutputWriter}()
   callbacks = OrderedDict{Symbol, Callback}()

   callbacks[:stop_iteration_exceeded] = Callback(stop_iteration_exceeded)

   # Convert numbers to floating point; otherwise preserve type (eg for DateTime types)
   #    TODO: implement TT = timetype(model) and FT = eltype(model)
   TT = eltype(model)
   Δt = Δt isa Number ? TT(Δt) : Δt

   return Simulation(model,
                     Δt,
                     Float64(stop_iteration),
                     nothing, # disallow stop_time
                     Float64(wall_time_limit),
                     diagnostics,
                     output_writers,
                     callbacks,
                     0.0,
                     false,
                     false,
                     verbose,
                     Float64(minimum_relative_step))
end


