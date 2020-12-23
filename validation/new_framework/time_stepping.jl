using OffsetArrays

@inline δx(i, c) = @inbounds c[i+1] - c[i]

one_time_step!(i, c, F, F₋₁, Δx, Δt, ::ForwardEuler)    = c[i] - Δt/Δx*(     δx(i, F)              )
one_time_step!(i, c, F, F₋₁, Δx, Δt, ::AdamsBashforth2) = c[i] - Δt/Δx*( 3 * δx(i, F) - δx(i, F₋₁) )/2

time_steppers = (
    ForwardEuler,
    AdamsBashforth2
)

"""

Advance the solution by one time step.

This currently allows for two possible time-stepping schemes

       Method             Accuracy
       ======             ========
    1) ForwardEuler      (first-order)
    2) AdamsBashforth2   (second-order)

Note: Solution has indicies starting at 0 or less, depending on the halo size.

"""

function update_solution(c, U, W, Δt, grid, scheme, time_stepper)

    c₋₁ = c.(grid.xC, grid.yC[1], grid.zC[1], -Δt, U, W);
    c₀  = c.(grid.xC, grid.yC[1], grid.zC[1],  0, U, W);
    
    F₀ = zeros(grid.Nx+1)
    F₋₁ = zeros(grid.Nx+1)

    for i in 1:grid.Nx+1, j in 1:grid.Ny, k in 1:grid.Nz
        
        F₀[i, j, k] = advective_tracer_flux_x(i, j, k, grid, scheme, U, c₀)
        F₋₁[i, j, k] = advective_tracer_flux_x(i, j, k, grid, scheme, U, c₋₁)
        #F₀[i, j, k] =  advective_flux(i, j, k, grid, scheme, U, c₀)
        #F₋₁[i ,j, k] = advective_flux(i, j, k, grid, scheme, U, c₋₁)
        
    end
    
    cₛᵢₘ = OffsetArray(zeros(grid.Nx+2), -1)

    for i in 1:grid.Nx
        cₛᵢₘ[i] = one_time_step!(i, c₀, F₀, F₋₁, grid.Δx, Δt, time_stepper())
    end
    
    return cₛᵢₘ
    
end

