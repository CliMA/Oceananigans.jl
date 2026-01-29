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

for n = 1:3
    time_step!(model, 0.03)
    @show model.velocities.u[1, 1, 1]
end
