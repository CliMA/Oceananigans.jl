module Seapickle
  export
    NumType,
    ρ,
    div_flux,
    u_dot_u,
    u_dot_v,
    u_dot_w,
    solve_for_pressure,

    αᵥ,
    T₀,
    S₀,
    p₀,
    βᵖ,
    βˢ,
    βᵀ,
    ρ₀,

    δˣ,
    δʸ,
    δᶻ

  using
    FFTW

  include("constants.jl")
  include("operators.jl")
  include("equation_of_state.jl")
  include("pressure_solve.jl")
  #include("generate_initial_conditions.jl")

end # module
