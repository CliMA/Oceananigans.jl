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
path = "https://github.com/simone-silvestri/OceananigansArtifacts.jl/raw/refs/heads/ss/new-data-for-regression-2/data_for_regression_tests/"

# DataDeps caches by name and never re-downloads while the cache directory exists. CI agents keep a
# persistent depot, so the version suffix must be bumped whenever the reference data is regenerated.
dh = DataDep("regression_truth_data_v2",
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

# A download that fails partway leaves the cache poisoned
reference_filenames = basename.(dh.remotepath)
datadep_path = DataDeps.try_determine_load_path("regression_truth_data_v2", pwd())

if datadep_path !== nothing && !(isdir(datadep_path) && all(f -> isfile(joinpath(datadep_path, f)), reference_filenames))
    @info "Discarding incomplete regression truth data at $datadep_path"
    rm(datadep_path; force=true, recursive=true)
end

datadep"regression_truth_data_v2"
