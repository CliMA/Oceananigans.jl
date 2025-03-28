
deconcretize(z::StaticVerticalDiscretization) =
    StaticVerticalDiscretization(
        deconcretize(z.cᵃᵃᶠ),
        deconcretize(z.cᵃᵃᶜ),
        deconcretize(z.Δᵃᵃᶠ),
        deconcretize(z.Δᵃᵃᶜ)
    )

# TODO: handle MutableVerticalDiscretization in grid constructors
deconcretize(z::MutableVerticalDiscretization) = z

function materialize_immersed_boundary(grid::ReactantGrid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    new_ib = GridFittedBottom(bottom_field)
    return new_ib
end

deconcretize(gfb::GridFittedBottom) = GridFittedBottom(deconcretize(gfb.bottom_height),
                                                       gfb.immersed_condition)

const CPUUnderlyingGrid{FT, TX, TY, TZ, CZ} = AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, <:CPU}
const CPUImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S} =
    ImmersedBoundaryGrid{FT, TX, TY, TZ, G, I, M, S, <:CPU}

function constant_with_arch(cpu_grid::AbstractUnderlyingGrid, arch)
    GridType = typeof(cpu_grid).name.wrapper
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
