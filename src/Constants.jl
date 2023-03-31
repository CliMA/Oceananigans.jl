module Constants

export R_Earth, Ω_Earth, g_Earth

using ..Units

"""
    const R_Earth
    
Mean radius of the Earth (in meters); see https://en.wikipedia.org/wiki/Earth
"""
const R_Earth = 6371kilometers

"""
    const Ω_Earth
    
Earth's rotation rate (in radians per second) which is equal to
`2π / (1 sideral day) ≈ 2π / (23hours + 56minutes + 4.1seconds)`;
see https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed
"""
const Ω_Earth = 7.292115e-5

"""
    const g_Earth

Gravitational acceleration at the Earth's surface (in meters / second²);
see https://en.wikipedia.org/wiki/Gravitational_acceleration#Gravity_model_for_Earth
"""
const g_Earth = 9.80665

end # module
