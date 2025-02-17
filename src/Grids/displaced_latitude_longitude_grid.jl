struct LatitudeLongitudeDisplacement{FT}
    north_pole :: FT
end

const DisplacedLatitudeLongitudeGrid = OrthogonalSphericalShellGrid{<:Any,
                                                                    <:Any,
                                                                    <:Any,
                                                                    <:Any,
                                                                    <:Any,
                                                                    <:Any,
                                                                    <:LatitudeLongitudeDisplacement}


function DisplacedLatitudeLongitudeGrid(architecture::AbstractArchitecture = CPU(),
                                        FT::DataType = Oceananigans.defaults.FloatType;
                                        size,
                                        north_pole,
                                        latitude = nothing,
                                        kw...)

    Δφ = 90 - north_pole
    shifted_latitude = latitude .+ Δφ
    lat_lon_grid = LatitudeLongitudeGrid(architecture, FT; size, latitude=shifted_latitude, kw...)
    Nx, Ny, Nz = size(lat_lon_grid)
    Hx, Hy, Hz = halo_size(lat_lon_grid)
    Lz = lat_lon_grid.Lz

    return OrthogonalSphericalShellGrid(architecture,
                                        Nx, Ny, Nz,
                                        Hx, Hy, Hz,
                                        Lz,
                                        λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                        φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, z,
                                        Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                        Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, 
                                        Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                        radius, conformal_mapping)

end
