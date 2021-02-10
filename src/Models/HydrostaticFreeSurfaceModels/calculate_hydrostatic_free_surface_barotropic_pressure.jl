import Oceananigans.TimeSteppers: calculate_pressure_correction!, pressure_correct_velocities!

calculate_pressure_correction!(::HydrostaticFreeSurfaceModel, Δt) = nothing
pressure_correct_velocities!(::HydrostaticFreeSurfaceModel, Δt) = nothing
