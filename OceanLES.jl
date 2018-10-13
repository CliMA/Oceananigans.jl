# ### Defining physical constants.
Ω = 7.2921150e-5  # Rotation rate of the Earth [rad/s].
f = 1e-4  # Nominal value for the Coriolis frequency [rad/s].
g = 9.80665  # Standard acceleration due to gravity [m/s²].
αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].

# ### Defining model parameters.
NumType = Float64  # Number data type.

Nˣ, Nʸ, Nᶻ = 100, 100, 50  # Number of grid points in (x,y,z).
Lˣ, Lʸ, Lᶻ = 2000, 2000, 1000  # Domain size [m].
Δx, Δy, Δz = Lˣ/Nˣ, Lʸ/Nʸ, Lᶻ/Nᶻ  # Grid spacing [m].

Nᵗ = 10  # Number of time steps to run for.
Δt = 20  # Time step [s].


@info "Ocean LES model starting up with parameters" Nˣ Nʸ Nᶻ NType

# ### Initializing prognostic and diagnostic variable fields.
u = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in x-direction [m/s].
v = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in y-direction [m/s].
w = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Velocity in z-direction [m/s].
θ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Potential temperature [K].
S = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Salinity [g/kg].
ρ = Array{NumType, 3}(undef, Nˣ, Nʸ, Nᶻ)  # Density [kg/m³].
