using DataDeps

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

dd = DataDep("cubed_sphere_32_grid",
             "Conformal cubed sphere grid with 32×32 grid points on each face",
             "https://github.com/glwagner/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cs32_with_4_halos/cubed_sphere_32_grid_with_4_halos.jld2")

DataDeps.register(dd)

# Trigger datadep download to avoid race condition in CI.
# See: https://github.com/oxinabox/DataDeps.jl/issues/141
datadep"cubed_sphere_32_grid"

# Downloading the regression fields
path = "https://github.com/simone-silvestri/OceananigansArtifacts.jl/raw/refs/heads/ss/new-data-for-regression-2/data_for_regression_tests"

dh = DataDep("regression_truth_data",
    "Data for Regression tests",
    [path * "hydrostatic_free_turbulence_regression_Periodic_ImplicitFreeSurface.jld2",
     path * "hydrostatic_free_turbulence_regression_Periodic_ExplicitFreeSurface.jld2",
     path * "hydrostatic_free_turbulence_regression_Periodic_SplitExplicitFreeSurface.jld2",
     path * "hydrostatic_free_turbulence_regression_Bounded_ImplicitFreeSurface.jld2",
     path * "hydrostatic_free_turbulence_regression_Bounded_ExplicitFreeSurface.jld2",
     path * "hydrostatic_free_turbulence_regression_Bounded_SplitExplicitFreeSurface.jld2",
     path * "hydrostatic_rotation_regression_Static_Nothing_AB2.jld2",
     path * "hydrostatic_rotation_regression_Static_CATKE_AB2.jld2",
     path * "hydrostatic_rotation_regression_Static_Nothing_RK3.jld2",
     path * "hydrostatic_rotation_regression_Static_CATKE_RK3.jld2",
     path * "hydrostatic_rotation_regression_Mutable_Nothing_AB2.jld2",
     path * "hydrostatic_rotation_regression_Mutable_CATKE_AB2.jld2",
     path * "hydrostatic_rotation_regression_Mutable_Nothing_RK3.jld2",
     path * "hydrostatic_rotation_regression_Mutable_CATKE_RK3.jld2",
     path * "ocean_large_eddy_simulation_AnisotropicMinimumDissipation_iteration10000.jld2",
     path * "ocean_large_eddy_simulation_AnisotropicMinimumDissipation_iteration10010.jld2",
     path * "ocean_large_eddy_simulation_SmagorinskyLilly_iteration10000.jld2",
     path * "ocean_large_eddy_simulation_SmagorinskyLilly_iteration10010.jld2",
     path * "ocean_large_eddy_simulation_DynamicSmagorinsky_directional_iteration10000.jld2",
     path * "ocean_large_eddy_simulation_DynamicSmagorinsky_directional_iteration10010.jld2",
     path * "ocean_large_eddy_simulation_DynamicSmagorinsky_lagrangian_iteration10000.jld2",
     path * "ocean_large_eddy_simulation_DynamicSmagorinsky_lagrangian_iteration10010.jld2",
     path * "rayleigh_benard_iteration1000.jld2",
     path * "rayleigh_benard_iteration1100.jld2",
     path * "thermal_bubble_regression.nc"]
)

DataDeps.register(dh)

# Invalidate stale DataDeps cache when any expected reference file is missing
# or has an unexpected size. Persistent CI caches (e.g. buildkite agents)
# may hold reference data from previous artifact uploads; expected sizes are
# the authoritative way to detect that.
#
# Sizes refer to the current files in glwagner/OceananigansArtifacts.jl
# (data_for_regression_tests/). When regenerating reference data, update
# these alongside the new upload.
const _expected_reference_sizes = Dict(
    "ocean_large_eddy_simulation_DynamicSmagorinsky_directional_iteration10000.jld2" => 713110,
    "ocean_large_eddy_simulation_DynamicSmagorinsky_directional_iteration10010.jld2" => 713110,
    "ocean_large_eddy_simulation_DynamicSmagorinsky_lagrangian_iteration10000.jld2"  => 1244676,
    "ocean_large_eddy_simulation_DynamicSmagorinsky_lagrangian_iteration10010.jld2"  => 1244676,
)

dd_path = try; datadep"regression_truth_data"; catch; nothing; end
if dd_path !== nothing
    cache_stale = any(_expected_reference_sizes) do (filename, expected_size)
        path = joinpath(dd_path, filename)
        !isfile(path) || filesize(path) != expected_size
    end
    if cache_stale
        @info "Regression truth data cache is stale, re-downloading..."
        rm(dd_path; recursive=true)
        datadep"regression_truth_data"
    end
else
    datadep"regression_truth_data"
end
