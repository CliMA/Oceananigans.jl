### Difference Operator

@inline δx(i, c) = @inbounds c[i+1] - c[i]

### Time Stepping Schemes

struct ForwardEuler end
struct AdamsBashforth2 end

Time_Stepper(i, c, F, F₋₁, Δx, Δt, ::ForwardEuler)    = c[i] - Δt/Δx*(     δx(i, F)              )
Time_Stepper(i, c, F, F₋₁, Δx, Δt, ::AdamsBashforth2) = c[i] - Δt/Δx*( 3 * δx(i, F) - δx(i, F₋₁) )/2

time_steppers = (
    ForwardEuler,
    AdamsBashforth2
)

