using Downloads

bathymetry_path = joinpath(@__DIR__, "bathymetry-360x150-latitude-75.0.jld2")
boundary_conditions_path = joinpath(@__DIR__, "boundary_conditions-1degree.jld2")

# TODO: convert to DataDeps

download_bathymetry(path=bathymetry_path) =
    Downloads.download("https://www.dropbox.com/s/axyzt88g0nr9dbc/bathymetry-360x150-latitude-75.0.jld2", bathymetry_path)

download_boundary_conditions(path=boundary_conditions_path) =
    Downloads.download("https://www.dropbox.com/s/7sbrq5fcnuhtvqp/boundary_conditions-1degree.jld2", path)

