"""
    ParameterizedForcing(func, parameters)

Construct a forcing function with parameters. The forcing function, which is
applied at grid point `i, j, k`, is called with the signature

    `func(i, j, k, grid, clock, state, parameters)`
    
Example
=======

function cool_forcing_function(i, j, k, grid, clock, state, parameters) = 
    return @inbounds - parameters.μ * exp(grid.zC[k] / parameters.λ) * state.velocities.u[i, j, k]
end

cool_forcing = ParameterizedForcing(cool_forcing_function, parameters=(μ=42, λ=π))
"""
struct ParameterizedForcing{F, P}
    func :: F
    parameters :: P
end

@inline (F::ParameterizedForcing)(i, j, k, grid, clock, state) = 
    F.func(i, j, k, grid, clock, state, F.parameters)
