# Minimal working example for PrescribedVelocityFields with FieldTimeSeries
#
# This script demonstrates using a FieldTimeSeries as input to PrescribedVelocityFields.
# The velocities are automatically interpolated to the current clock time during simulation.

using Oceananigans

# Create grid and time series
grid = RectilinearGrid(size=(4, 4, 4), extent=(1, 1, 1))
times = 0:0.1:1.0

# Create velocity FieldTimeSeries and populate with set!
u_fts = FieldTimeSeries{Face, Center, Center}(grid, times)
set!(u_fts, (x, y, z, t) -> t)

# Use with PrescribedVelocityFields
velocities = PrescribedVelocityFields(; u=u_fts)
model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c)

simulation = Simulation(
    model;
    Δt = 0.05,
    stop_time = 1.0,
)

output_fields = Dict(
    "u" => model.velocities.u,
)

filename = "prescribed_velocity_field_time_series_mwe.jld2"

simulation.output_writers[:fields] = JLD2Writer(
    model, output_fields;
    schedule = TimeInterval(0.1),
    filename = filename,
    overwrite_existing = true,
)

run!(simulation)

u_fts_output = FieldTimeSeries(filename, "u")

for n in eachindex(u_fts_output.times)
    u = u_fts_output[n]
    @show u[1, 1, 1]
end
