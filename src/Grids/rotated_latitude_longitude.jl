
struct RotatedLatitudeLongitude{N}
    north_pole::N
end

Adapt.adapt_structure(to, t::RotatedLatitudeLongitude) = 
    RotatedLatitudeLongitude(Adapt.adapt(to, t.north_pole))

@inline function convert_to_minus90_90(x) 
    m = (((x + 90) % 180) + 180) % 180 - 90
    n = floor((x + 90) / 180)
    if iseven(n)
        return m
    else
        return -m
    end
end

@inline convert_to_0_360(x) = ((x % 360) + 360) % 360

function RotatedLatitudeLongitudeGrid(architecture::AbstractArchitecture = CPU(),
                                      FT::DataType = Float64;
                                      size,
                                      longitude = nothing,
                                      latitude = nothing,
                                      z = nothing,
                                      radius = R_Earth,
                                      north_pole = (0, 90),
                                      topology = nothing,
                                      halo = nothing)

    latitude_longitude_grid = LatitudeLongitudeGrid(architecture, FT;
                                                    size,
                                                    longitude,
                                                    latitude,
                                                    z,
                                                    radius,
                                                    topology,
                                                    halo)

    TX, TY, TZ = Grids.topology(latitude_longitude_grid)
    conformal_mapping = RotatedLatitudeLongitude(north_pole)

    φᶜᶜᵃ = new_data(FT, architecture, (Center, Center, Nothing), topology, size, halo)
    φᶠᶜᵃ = new_data(FT, architecture, (Face, Center, Nothing),   topology, size, halo)
    φᶜᶠᵃ = new_data(FT, architecture, (Center, Face, Nothing),   topology, size, halo)
    φᶠᶠᵃ = new_data(FT, architecture, (Face, Face, Nothing),     topology, size, halo)

    λᶜᶜᵃ = new_data(FT, architecture, (Center, Center, Nothing), topology, size, halo)
    λᶠᶜᵃ = new_data(FT, architecture, (Face, Center, Nothing),   topology, size, halo)
    λᶜᶠᵃ = new_data(FT, architecture, (Center, Face, Nothing),   topology, size, halo)
    λᶠᶠᵃ = new_data(FT, architecture, (Face, Face, Nothing),     topology, size, halo)

    

    arch = architecture

    φᶜᶜᵃ = latitude_longitude_grid.φᵃᶜᵃ .+ north_pole[2]
    φᶠᶜᵃ = latitude_longitude_grid.φᵃᶜᵃ .+ north_pole[2]
    φᶜᶠᵃ = latitude_longitude_grid.φᵃᶠᵃ .+ north_pole[2]
    φᶠᶠᵃ = latitude_longitude_grid.φᵃᶠᵃ .+ north_pole[2]

    λᶜᶜᵃ = latitude_longitude_grid.λᶜᵃᵃ .+ north_pole[1]
    λᶠᶜᵃ = latitude_longitude_grid.λᶜᵃᵃ .+ north_pole[1]
    λᶜᶠᵃ = latitude_longitude_grid.λᶠᵃᵃ .+ north_pole[1]
    λᶠᶠᵃ = latitude_longitude_grid.λᶠᵃᵃ .+ north_pole[1]

    φᶜᶜᵃ = convert_to_minus90_90.(φᶜᶜᵃ)
    φᶠᶜᵃ = convert_to_minus90_90.(φᶠᶜᵃ)
    φᶜᶠᵃ = convert_to_minus90_90.(φᶜᶠᵃ)
    φᶠᶠᵃ = convert_to_minus90_90.(φᶠᶠᵃ)

    λᶜᶜᵃ = convert_to_0_360.(λᶜᶜᵃ)
    λᶠᶜᵃ = convert_to_0_360.(λᶠᶜᵃ)
    λᶜᶠᵃ = convert_to_0_360.(λᶜᶠᵃ)
    λᶠᶠᵃ = convert_to_0_360.(λᶠᶠᵃ)
    
    grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(arch,
                                                    Grids.size(latitude_longitude_grid)...,
                                                    Grids.halo_size(latitude_longitude_grid)...,
                                                    convert(eltype(radius), latitude_longitude_grid.Lz),
                                                    on_architecture(arch, λᶜᶜᵃ),
                                                    on_architecture(arch, λᶠᶜᵃ),
                                                    on_architecture(arch, λᶜᶠᵃ),
                                                    on_architecture(arch, λᶠᶠᵃ),
                                                    on_architecture(arch, φᶜᶜᵃ),
                                                    on_architecture(arch, φᶠᶜᵃ),
                                                    on_architecture(arch, φᶜᶠᵃ),
                                                    on_architecture(arch, φᶠᶠᵃ),
                                                    on_architecture(arch, latitude_longitude_grid.zᵃᵃᶜ),
                                                    on_architecture(arch, latitude_longitude_grid.zᵃᵃᶠ),
                                                    on_architecture(arch, Δxᶜᶜᵃ),
                                                    on_architecture(arch, Δxᶠᶜᵃ),
                                                    on_architecture(arch, Δxᶜᶠᵃ),
                                                    on_architecture(arch, Δxᶠᶠᵃ),
                                                    on_architecture(arch, Δyᶜᶜᵃ),
                                                    on_architecture(arch, Δyᶜᶠᵃ),
                                                    on_architecture(arch, Δyᶠᶜᵃ),
                                                    on_architecture(arch, Δyᶠᶠᵃ),
                                                    on_architecture(arch, latitude_longitude_grid.Δzᵃᵃᶜ),
                                                    on_architecture(arch, latitude_longitude_grid.Δzᵃᵃᶠ),
                                                    on_architecture(arch, Azᶜᶜᵃ),
                                                    on_architecture(arch, Azᶠᶜᵃ),
                                                    on_architecture(arch, Azᶜᶠᵃ),
                                                    on_architecture(arch, Azᶠᶠᵃ),
                                                    radius,
                                                    conformal_mapping)

    return grid
end

