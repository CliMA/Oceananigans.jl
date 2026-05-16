function vertical_spacing_field(grid::AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:GPU}, lz)
    field = Field{Nothing, Nothing, typeof(lz)}(grid)
    Δ = lz isa Center ? grid.z.Δᵃᵃᶜ : grid.z.Δᵃᵃᶠ
    Nz_int = length(interior_indices(lz, topology(grid, 3)(), size(grid, 3)))

    interior_data = Δ isa Number ? fill(eltype(grid)(Δ), Nz_int) :
                                   eltype(grid).(collect(on_architecture(CPU(), Δ)[1:Nz_int]))

    set!(field, reshape(interior_data, (1, 1, Nz_int)))

    return field
end
