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
