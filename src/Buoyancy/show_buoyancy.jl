using Printf

Base.show(io::IO, b::SeawaterBuoyancy{FT}) where FT =
    println(io, "SeawaterBuoyancy{$FT}: g = $(b.gravitational_acceleration)", '\n',
                "└── equation of state: $(b.equation_of_state)")

Base.show(io::IO, eos::LinearEquationOfState{FT}) where FT =
    println(io, "LinearEquationOfState{$FT}: ", @sprintf("α = %.2e, β = %.2e", eos.α, eos.β))
