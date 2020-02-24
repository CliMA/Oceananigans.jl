using GPUifyLoops: @launch, @loop
using Oceananigans.Operators: Δy, ΔzF
using Oceananigans.Utils: @loop_xy, @loop_xz, launch_config

#####
##### Algorithm for adding fluxes associated with non-trivial flux boundary conditions.
##### Inhomogeneous Value and Gradient boundary conditions are handled by filling halos.
#####

"""
    apply_z_bcs!(Gc, arch, grid, args...)

Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the top and bottom.
"""
function apply_z_bcs!(Gc, c, arch, args...)
    @launch(device(arch), config=launch_config(Gc.grid, :xy),
            _apply_z_bcs!(Gc.data, Gc.grid, c.boundary_conditions.z.bottom,
                          c.boundary_conditions.z.top, args...))
    return nothing
end

"""
    apply_y_bcs!(Gc, arch, grid, args...)

Apply flux boundary conditions to a field `c` by adding the associated flux divergence to
the source term `Gc` at the left and right.
"""
function apply_y_bcs!(Gc, c, arch, args...)
    @launch(device(arch), config=launch_config(Gc.grid, :xz),
            _apply_y_bcs!(Gc.data, Gc.grid, c.boundary_conditions.y.left,
                          c.boundary_conditions.y.right, args...))
    return nothing
end

"""
    _apply_z_bcs!(Gc, grid, bottom_bc, top_bc, args...)

Apply a top and/or bottom boundary condition to variable `c`.
"""
function _apply_z_bcs!(Gc, grid, bottom_bc, top_bc, args...)
    @loop_xy i j grid begin
        apply_z_bottom_bc!(Gc, bottom_bc, i, j, grid, args...)
           apply_z_top_bc!(Gc, top_bc,    i, j, grid, args...)
    end
end

"""
    _apply_z_bcs!(Gc, grid, left_bc, right_bc, args...)

Apply a left and/or right boundary condition to variable `c`.
"""
function _apply_y_bcs!(Gc, grid, left_bc, right_bc, args...)
    @loop_xz i k grid begin
         apply_y_left_bc!(Gc, left_bc,  i, k, grid, args...)
        apply_y_right_bc!(Gc, right_bc, i, k, grid, args...)
    end
end

# Fall back functions for boundary conditions that are not of type Flux.
@inline apply_z_top_bc!(args...) = nothing
@inline apply_z_bottom_bc!(args...) = nothing

@inline apply_y_right_bc!(args...) = nothing
@inline apply_y_left_bc!(args...) = nothing

# Shortcuts for 'no-flux' boundary conditions.
@inline apply_z_top_bc!(Gc, ::NFBC, args...) = nothing
@inline apply_z_bottom_bc!(Gc, ::NFBC, args...) = nothing

@inline apply_y_right_bc!(Gc, ::NFBC, args...) = nothing
@inline apply_y_left_bc!(Gc, ::NFBC, args...) = nothing

"""
    apply_z_top_bc!(Gc, top_flux::BC{<:Flux}, i, j, grid, args...)

Add the part of flux divergence associated with a top boundary condition on `c`.
Note that because

    `tendency = ∂c/∂t = Gc = - ∇ ⋅ flux`

a positive top flux is associated with a *decrease* in `Gc` near the top boundary.
If `top_bc.condition` is a function, the function must have the signature

    `top_bc.condition(i, j, grid, boundary_condition_args...)`
"""
@inline apply_z_top_bc!(Gc, top_flux::BC{<:Flux}, i, j, grid, args...) =
    @inbounds Gc[i, j, grid.Nz] -= getbc(top_flux, i, j, grid, args...) / ΔzF(i, j, grid.Nz, grid)

@inline apply_y_right_bc!(Gc, right_flux::BC{<:Flux}, i, k, grid, args...) =
    @inbounds Gc[i, grid.Ny, k] -= getbc(right_flux, i, k, grid, args...) / Δy(i, grid.Ny, k, grid)

"""
    apply_z_bottom_bc!(Gc, bottom_flux::BC{<:Flux}, i, j, grid, args...)

Add the flux divergence associated with a bottom flux boundary condition on `c`.
Note that because

    `tendency = ∂c/∂t = Gc = - ∇ ⋅ flux`

a positive bottom flux is associated with an *increase* in `Gc` near the bottom boundary.
If `bottom_bc.condition` is a function, the function must have the signature

    `bottom_bc.condition(i, j, grid, boundary_condition_args...)`
"""
@inline apply_z_bottom_bc!(Gc, bottom_flux::BC{<:Flux}, i, j, grid, args...) =
    @inbounds Gc[i, j, 1] += getbc(bottom_flux, i, j, grid, args...) / ΔzF(i, j, 1, grid)

@inline apply_y_left_bc!(Gc, left_flux::BC{<:Flux}, i, k, grid, args...) =
    @inbounds Gc[i, 1, k] += getbc(left_flux, i, k, grid, args...) / Δy(i, 1, k, grid)
