using DataDeps

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2"
)

DataDeps.register(dd)


# Trigger datadep download to avoid race condition in CI.
# See: https://github.com/oxinabox/DataDeps.jl/issues/141
datadep"cubed_sphere_32_grid"

# Downloading the regression fields

path = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/data_for_regression_tests/"

dh = DataDep("regression_test_data",
    "Data for Regression tests",
    [path * "hydrostatic_free_turbulence_regression_Periodic_ImplicitFreeSurface.jld2",
     path * "hydrostatic_free_turbulence_regression_Periodic_ExplicitFreeSurface.jld2",
     path * "hydrostatic_free_turbulence_regression_Bounded_ImplicitFreeSurface.jld2",
     path * "hydrostatic_free_turbulence_regression_Bounded_ExplicitFreeSurface.jld2",
     path * "ocean_large_eddy_simulation_AnisotropicMinimumDissipation_iteration10000.jld2",
     path * "ocean_large_eddy_simulation_AnisotropicMinimumDissipation_iteration10010.jld2",
     path * "ocean_large_eddy_simulation_SmagorinskyLilly_iteration10000.jld2",
     path * "ocean_large_eddy_simulation_SmagorinskyLilly_iteration10010.jld2",
     path * "rayleigh_benard_iteration1000.jld2",
     path * "rayleigh_benard_iteration1100.jld2",
     path * "thermal_bubble_regression.nc"]
)

DataDeps.register(dh)

datadep"regression_test_data"