"""

Insert something here

"""

@inline δx(i, c) = @inbounds c[i+1] - c[i]

one_time_step!(i, c, F, F₋₁, Δx, Δt, ::ForwardEuler)    = c[i] - Δt/Δx*(     δx(i, F)              )
one_time_step!(i, c, F, F₋₁, Δx, Δt, ::AdamsBashforth2) = c[i] - Δt/Δx*( 3 * δx(i, F) - δx(i, F₋₁) )/2

time_steppers = (
    ForwardEuler,
    AdamsBashforth2
)

