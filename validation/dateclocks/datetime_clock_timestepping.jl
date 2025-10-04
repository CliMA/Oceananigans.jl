using Oceananigans
using Dates
using Oceananigans.Units: hour, Time
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Architectures: CPU

function run_datetime_clock_validation(; arch = CPU())
    grid = RectilinearGrid(arch, size=(1, 1, 1), extent=(1, 1, 1))
    start_time = DateTime(2020, 1, 1)
    times = start_time:Hour(1):start_time + Hour(3)

    u_forcing = FieldTimeSeries{Face, Center, Center}(grid, times)

    for (n, _) in enumerate(times)
        set!(u_forcing[n], (x, y, z) -> Float64(n - 1))
    end

    clock = Clock(time=start_time)
    model = HydrostaticFreeSurfaceModel(; grid, clock, forcing=(; u=u_forcing))

    forcing_history = Float64[]
    time_history = DateTime[]

    push!(forcing_history, u_forcing[1, 1, 1, Time(model.clock.time)])
    push!(time_history, model.clock.time)

    time_step!(model, hour; euler=true)
    push!(forcing_history, u_forcing[1, 1, 1, Time(model.clock.time)])
    push!(time_history, model.clock.time)

    for _ in 1:2
        time_step!(model, hour)
        push!(forcing_history, u_forcing[1, 1, 1, Time(model.clock.time)])
        push!(time_history, model.clock.time)
    end

    expected_times = [start_time + Hour(n) for n in 0:3]
    expected_forcing = Float64.(0:3)

    @assert time_history == expected_times "Clock time history does not match expected hourly dates."
    @assert forcing_history == expected_forcing "FieldTimeSeries forcing did not align with DateTime clock."

    return (; time_history, forcing_history)
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_datetime_clock_validation()
    @info "DateTime clock validation succeeded" results...
end
