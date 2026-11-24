using Oceananigans.Grids: ξnodes, ηnodes, rnodes

# Generic reconstruction methods valid for all reconstruction schemes
# Unroll the functions to pass the coordinates in case of a stretched grid
""" same as [`symmetric_interpolate_xᶠᵃᵃ`](@ref) but on `Center`s instead of `Face`s """
@inline symmetric_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, ψ, args...) = symmetric_interpolate_xᶠᵃᵃ(i + 1, j, k, grid, scheme, ψ, args...)
""" same as [`symmetric_interpolate_yᵃᶠᵃ`](@ref) but on `Center`s instead of `Face`s """
@inline symmetric_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, ψ, args...) = symmetric_interpolate_yᵃᶠᵃ(i, j + 1, k, grid, scheme, ψ, args...)
""" same as [`symmetric_interpolate_zᵃᵃᶠ`](@ref) but on `Center`s instead of `Face`s """
@inline symmetric_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, ψ, args...) = symmetric_interpolate_zᵃᵃᶠ(i, j, k + 1, grid, scheme, ψ, args...)

""" same as [`biased_interpolate_xᶠᵃᵃ`](@ref) but on `Center`s instead of `Face`s """
@inline biased_interpolate_xᶜᵃᵃ(i, j, k, grid, scheme, bias, ψ, args...)  = biased_interpolate_xᶠᵃᵃ(i + 1, j, k, grid, scheme, bias, ψ, args...)
""" same as [`biased_interpolate_yᵃᶠᵃ`](@ref) but on `Center`s instead of `Face`s """
@inline biased_interpolate_yᵃᶜᵃ(i, j, k, grid, scheme, bias, ψ, args...)  = biased_interpolate_yᵃᶠᵃ(i, j + 1, k, grid, scheme, bias, ψ, args...)
""" same as [`biased_interpolate_zᵃᵃᶠ`](@ref) but on `Center`s instead of `Face`s """
@inline biased_interpolate_zᵃᵃᶜ(i, j, k, grid, scheme, bias, ψ, args...)  = biased_interpolate_zᵃᵃᶠ(i, j, k + 1, grid, scheme, bias, ψ, args...)

#####
##### Shenanigans for stretched directions
#####

@inline function reconstruction_stencil(buffer, shift, dir, func::Bool = false)
    N = buffer * 2
    order = shift == :symmetric ? N : N - 1
    if shift != :symmetric
        N = N .- 1
    end
    rng = 1:N
    if shift == :right
        rng = rng .+ 1
    end
    stencil_full = Vector(undef, N)
    for (idx, n) in enumerate(rng)
        c = n - buffer - 1
        if func
            stencil_full[idx] = dir == :x ?
                                :(ψ(i + $c, j, k, grid, args...)) :
                                dir == :y ?
                                :(ψ(i, j + $c, k, grid, args...)) :
                                :(ψ(i, j, k + $c, grid, args...))
        else
            stencil_full[idx] = dir == :x ?
                                :(ψ[i + $c, j, k]) :
                                dir == :y ?
                                :(ψ[i, j + $c, k]) :
                                :(ψ[i, j, k + $c])
        end
    end
    return :($(reverse(stencil_full)...),)
end

@inline function compute_reconstruction_coefficients(grid, FT, scheme; order)

    method = scheme == :Centered ? 1 : scheme == :Upwind ? 2 : 3

    if grid isa Nothing
        coeff_xᶠᵃᵃ = nothing
        coeff_xᶜᵃᵃ = nothing
        coeff_yᵃᶠᵃ = nothing
        coeff_yᵃᶜᵃ = nothing
        coeff_zᵃᵃᶠ = nothing
        coeff_zᵃᵃᶜ = nothing
    else
        arch       = architecture(grid)
        Hx, Hy, Hz = halo_size(grid)
        new_grid   = with_halo((Hx+1, Hy+1, Hz+1), grid)

        ξᶠᵃᵃ = ξnodes(new_grid, Face(), with_halos=true)
        ηᵃᶠᵃ = ηnodes(new_grid, Face(), with_halos=true)
        rᵃᵃᶠ = rnodes(new_grid, Face(), with_halos=true)

        ξᶜᵃᵃ = ξnodes(new_grid, Center(), with_halos=true)
        ηᵃᶜᵃ = ηnodes(new_grid, Center(), with_halos=true)
        rᵃᵃᶜ = rnodes(new_grid, Center(), with_halos=true)

        coeff_xᶠᵃᵃ = reconstruction_coefficients(FT, ξᶠᵃᵃ, arch, new_grid.Nx, Val(method); order)
        coeff_xᶜᵃᵃ = reconstruction_coefficients(FT, ξᶜᵃᵃ, arch, new_grid.Nx, Val(method); order)
        coeff_yᵃᶠᵃ = reconstruction_coefficients(FT, ηᵃᶠᵃ, arch, new_grid.Ny, Val(method); order)
        coeff_yᵃᶜᵃ = reconstruction_coefficients(FT, ηᵃᶜᵃ, arch, new_grid.Ny, Val(method); order)
        coeff_zᵃᵃᶠ = reconstruction_coefficients(FT, rᵃᵃᶠ, arch, new_grid.Nz, Val(method); order)
        coeff_zᵃᵃᶜ = reconstruction_coefficients(FT, rᵃᵃᶜ, arch, new_grid.Nz, Val(method); order)
    end

    return (coeff_xᶠᵃᵃ, coeff_xᶜᵃᵃ, coeff_yᵃᶠᵃ, coeff_yᵃᶜᵃ, coeff_zᵃᵃᶠ, coeff_zᵃᵃᶜ)
end

# Fallbacks for uniform or Flat directions
for val in [1, 2, 3]
    @eval begin
        @inline reconstruction_coefficients(FT, coord::StaticStridesOffsetArray{<:Any, <:Any, <:Any, <:Any, <:AbstractRange}, arch, N, ::Val{$val}; order) = nothing
        @inline reconstruction_coefficients(FT, coord::AbstractRange, arch, N, ::Val{$val}; order)                              = nothing
        @inline reconstruction_coefficients(FT, coord::Nothing, arch, N, ::Val{$val}; order)                                    = nothing
        @inline reconstruction_coefficients(FT, coord::Number, arch, N, ::Val{$val}; order)                                     = nothing
    end
end

# Stretched reconstruction coefficients for `Centered` schemes
@inline function reconstruction_coefficients(FT, coord, arch, N, ::Val{1}; order)
    cpu_coord = on_architecture(CPU(), coord)
    r = ((order + 1) ÷ 2) - 1
    s = create_reconstruction_coefficients(FT, r, cpu_coord, arch, N; order)
    return s
end

# Stretched reconstruction coefficients for `UpwindBiased` schemes
@inline function reconstruction_coefficients(FT, coord, arch, N, ::Val{2}; order)
    cpu_coord = on_architecture(CPU(), coord)
    rleft  = ((order + 1) ÷ 2) - 2
    rright = ((order + 1) ÷ 2) - 1
    s = []
    for r in [rleft, rright]
        push!(s, create_reconstruction_coefficients(FT, r, cpu_coord, arch, N; order))
    end
    return tuple(s...)
end

# Stretched reconstruction coefficients for `WENO` schemes
@inline function reconstruction_coefficients(FT, coord, arch, N, ::Val{3}; order)
    cpu_coord = on_architecture(CPU(), coord)
    s = []
    for r in -1:order-1
        push!(s, create_reconstruction_coefficients(FT, r, cpu_coord, arch, N; order))
    end
    return tuple(s...)
end

# general reconstruction coefficients for order `order` and stencil `r` where r
@inline function create_reconstruction_coefficients(FT, r, cpu_coord, arch, N; order)
    stencil = NTuple{order, FT}[]
    @inbounds begin
        for i = 0:N+1
            push!(stencil, stencil_coefficients(FT, i, r, cpu_coord, cpu_coord; order))
        end
    end
    return StaticStridesOffsetArray(on_architecture(arch, stencil), -1)
end
