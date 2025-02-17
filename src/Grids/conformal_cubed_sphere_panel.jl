struct CubedSphereConformalMapping{FT, Rotation}
    ξ :: Tuple{FT, FT}
    η :: Tuple{FT, FT}
    rotation :: Rotation
end

const ConformalCubedSpherePanelGrid = OrthogonalSphericalShellGrid{<:Any,
                                                                   <:Any,
                                                                   <:Any,
                                                                   <:Any,
                                                                   <:Any,
                                                                   <:Any,
                                                                   <:CubedSphereConformalMapping}

# architecture = CPU() default, assuming that a DataType positional arg
# is specifying the floating point type.
conformal_cubed_sphere_panel(FT::DataType; kwargs...) = conformal_cubed_sphere_panel(CPU(), FT; kwargs...)

function load_and_offset_cubed_sphere_data(file, FT, arch, field_name, loc, topo, N, H)

    data = on_architecture(arch, file[field_name])
    data = convert.(FT, data)

    return offset_data(data, loc[1:2], topo[1:2], N[1:2], H[1:2])
end

function conformal_cubed_sphere_panel(filepath::AbstractString, architecture = CPU(), FT = Float64;
                                      panel, Nz, z,
                                      topology = (FullyConnected, FullyConnected, Bounded),
                                        radius = R_Earth,
                                          halo = (4, 4, 4),
                                      rotation = nothing)

    TX, TY, TZ = topology
    Hx, Hy, Hz = halo

    ## Read everything from the file except the z-coordinates

    file = jldopen(filepath, "r")["panel$panel"]

    Nξ, Nη = size(file["λᶠᶠᵃ"])
    Hξ, Hη = halo[1], halo[2]
    Nξ -= 2Hξ
    Nη -= 2Hη

    N = (Nξ, Nη, Nz)
    H = halo

    loc_cc = (Center, Center)
    loc_fc = (Face,   Center)
    loc_cf = (Center, Face)
    loc_ff = (Face,   Face)

     λᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "λᶜᶜᵃ", loc_cc, topology, N, H)
     λᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "λᶠᶠᵃ", loc_ff, topology, N, H)

     φᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "φᶜᶜᵃ", loc_cc, topology, N, H)
     φᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "φᶠᶠᵃ", loc_ff, topology, N, H)

    Δxᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶜᶜᵃ", loc_cc, topology, N, H)
    Δxᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶠᶜᵃ", loc_fc, topology, N, H)
    Δxᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶜᶠᵃ", loc_cf, topology, N, H)
    Δxᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δxᶠᶠᵃ", loc_ff, topology, N, H)

    Δyᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶜᶜᵃ", loc_cc, topology, N, H)
    Δyᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶠᶜᵃ", loc_fc, topology, N, H)
    Δyᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶜᶠᵃ", loc_cf, topology, N, H)
    Δyᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Δyᶠᶠᵃ", loc_ff, topology, N, H)

    Azᶜᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶜᶜᵃ", loc_cc, topology, N, H)
    Azᶠᶜᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶠᶜᵃ", loc_fc, topology, N, H)
    Azᶜᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶜᶠᵃ", loc_cf, topology, N, H)
    Azᶠᶠᵃ = load_and_offset_cubed_sphere_data(file, FT, architecture, "Azᶠᶠᵃ", loc_ff, topology, N, H)

    ## Maybe we won't need these?
    Txᶠᶜ = total_length(loc_fc[1](), topology[1](), N[1], H[1])
    Txᶜᶠ = total_length(loc_cf[1](), topology[1](), N[1], H[1])
    Tyᶠᶜ = total_length(loc_fc[2](), topology[2](), N[2], H[2])
    Tyᶜᶠ = total_length(loc_cf[2](), topology[2](), N[2], H[2])

    λᶠᶜᵃ = offset_data(zeros(architecture, FT, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology[1:2], N[1:2], H[1:2])
    λᶜᶠᵃ = offset_data(zeros(architecture, FT, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology[1:2], N[1:2], H[1:2])
    φᶠᶜᵃ = offset_data(zeros(architecture, FT, Txᶠᶜ, Tyᶠᶜ), loc_fc, topology[1:2], N[1:2], H[1:2])
    φᶜᶠᵃ = offset_data(zeros(architecture, FT, Txᶜᶠ, Tyᶜᶠ), loc_cf, topology[1:2], N[1:2], H[1:2])

    ## The vertical coordinates can come out of the regular rectilinear grid!
    Lz, z  = generate_coordinate(FT, topology, (Nξ, Nη, Nz), halo, z,  :z, 3, architecture)

    ξ, η = (-1, 1), (-1, 1)
    conformal_mapping = CubedSphereConformalMapping(ξ, η, rotation)

    return OrthogonalSphericalShellGrid{TX, TY, TZ}(architecture, Nξ, Nη, Nz, Hx, Hy, Hz, Lz,
                                                     λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                     φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ,
                                                     z,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ,
                                                    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                    radius,
                                                    conformal_mapping)
end

function with_halo(new_halo, old_grid::OrthogonalSphericalShellGrid; rotation=nothing)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)
    topo = topology(old_grid)

    ξ = old_grid.conformal_mapping.ξ
    η = old_grid.conformal_mapping.η

    z = cpu_face_constructor_z(old_grid)

    new_grid = conformal_cubed_sphere_panel(architecture(old_grid), eltype(old_grid);
                                            size, z, ξ, η,
                                            topology = topo,
                                            radius = old_grid.radius,
                                            halo = new_halo,
                                            rotation)

    return new_grid
end


