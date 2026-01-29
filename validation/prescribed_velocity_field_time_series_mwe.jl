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
for (n, t) in enumerate(times)
    set!(u_fts, t, n)  # u = t at each time index
end

# Use with PrescribedVelocityFields
velocities = PrescribedVelocityFields(; u=u_fts)
model = HydrostaticFreeSurfaceModel(grid; velocities, tracers=:c)

# At t=0, velocity field should interpolate to u=0
u = model.velocities.u
@show u[1, 1, 1]  # Should return 0.0

# Time step to t=0.05
time_step!(model, 0.05)

# Now u should interpolate to 0.05 (between t=0 and t=0.1)
@show u[1, 1, 1]  # Should return 0.05

# Time step further to t=0.15
time_step!(model, 0.10)

# Now u should interpolate to 0.15
@show u[1, 1, 1]  # Should return 0.15

# Verify the interpolation is working correctly
@assert u[1, 1, 1] ≈ 0.15 "Interpolation failed: expected 0.15, got $(u[1, 1, 1])"

println("Success! PrescribedVelocityFields with FieldTimeSeries works correctly.")
