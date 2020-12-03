"""
    gravity
A parameter object for gravitational acceleration in the vertical direction.
"""
struct Gravity{FT}
    g :: FT
end

"""
    Gravity([FT=Float64;] g=nothing)
Returns a parameter object for gravitational acceleration.
"""
function Gravity(FT::DataType=Float64; g=nothing)

    g = 9.81

end

Base.show(io::IO, gravity::Gravity{FT}) where FT =
    print(io, "Gravity{$FT}: g = ", @sprintf("%.2e", gravity.g))
