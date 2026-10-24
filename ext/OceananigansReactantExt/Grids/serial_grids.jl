
deconcretize(z::StaticVerticalDiscretization) =
    StaticVerticalDiscretization(
        deconcretize(z.cᵃᵃᶠ),
        deconcretize(z.cᵃᵃᶜ),
        deconcretize(z.Δᵃᵃᶠ),
        deconcretize(z.Δᵃᵃᶜ)
    )

# TODO: handle MutableVerticalDiscretization in grid constructors
deconcretize(z::MutableVerticalDiscretization) = z
deconcretize(gfb::GridFittedVerticalBoundaries) = GridFittedVerticalBoundaries(deconcretize(gfb.bottom_height),
                                                                               deconcretize(gfb.top_height),
                                                                               gfb.immersed_condition,
                                                                               deconcretize(gfb.bottom_cell_index),
                                                                               deconcretize(gfb.top_cell_index),
                                                                               deconcretize_face_column_depths(gfb.face_column_depths))

deconcretize_face_column_depths(::Nothing) = nothing
deconcretize_face_column_depths(depths) = map(deconcretize, depths)

const CPUUnderlyingGrid{FT, TX, TY, TZ, CZ} = AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, <:CPU}
const CPUImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S} =
    ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, <:CPU}

function constant_with_arch(cpu_grid::AbstractUnderlyingGrid, arch)
    GridType = unparameterized_type(typeof(cpu_grid))
    other_names = propertynames(cpu_grid)[2:end] # exclude architecture
    other_properties = Tuple(getproperty(cpu_grid, name) for name in other_names)
    TX, TY, TZ = Oceananigans.Grids.topology(cpu_grid)
    return GridType{TX, TY, TZ}(arch, other_properties...)
end

function constant_with_arch(cpu_ibg::CPUImmersedBoundaryGrid, arch)
    underlying_grid = constant_with_arch(cpu_ibg.underlying_grid, arch)
    TX, TY, TZ = Oceananigans.Grids.topology(cpu_ibg)
    return ImmersedBoundaryGrid{TX, TY, TZ}(underlying_grid,
                                            cpu_ibg.immersed_boundary,
                                            cpu_ibg.interior_active_cells,
                                            cpu_ibg.active_z_columns)
end
