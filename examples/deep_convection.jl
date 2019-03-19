using Statistics: mean
using Oceananigans

include("utils.jl")

function impose_cooling_disk!(model::Model)
    g = model.grid
    c = model.constants

    # Parameters for generating initial surface heat flux.
    Rc = 600   # Radius of cooling disk [m].
    Ts = 20    # Surface temperature [°C].
    Q₀ = -800  # Cooling disk heat flux [W/m²].
    Q₁ = 0     # Noise added to cooling disk heat flux [W/m²].
    Ns = 5 * (c.f * Rc/g.Lz)  # Stratification or Brunt–Väisälä frequency [s⁻¹].

    αᵥ = 2.07e-4  # Volumetric coefficient of thermal expansion for water [K⁻¹].
    cₚ = 4181.3   # Isobaric mass heat capacity [J / kg·K].

    Tz = Ns^2 / (c.g * αᵥ)  # Vertical temperature gradient [K/m].

    # Center horizontal coordinates so that (x,y) = (0,0) corresponds to the center
    # of the domain (and the cooling disk).
    x₀ = g.xC .- mean(g.xC)
    y₀ = g.yC .- mean(g.yC)

    # Calculate vertical temperature profile and convert to Kelvin.
    T_ref = 273.15 .+ Ts .+ Tz .* (g.zC .- mean(Tz * g.zC))

    # Impose reference temperature profile.
    model.tracers.T.data .= repeat(reshape(T_ref, 1, 1, g.Nz), g.Nx, g.Ny, 1)

    # Add small temperature perturbations to the surface.
    model.tracers.T.data[:, :, 1] .+= 0.001*rand()

    # Generate surface heat flux field.
    Q = Q₀ .+ Q₁ * (0.5 .+ rand(g.Nx, g.Ny))

    # Set surface heat flux to zero outside of cooling disk of radius Rc.
    r₀² = @. x₀*x₀ + y₀'*y₀'
    Q[findall(r₀² .> Rc^2)] .= 0

    # Convert surface heat flux into 3D forcing term for use when calculating
    # source terms at each time step. Also convert surface heat flux [W/m²]
    # into a temperature tendency forcing [K/s].
    # @. model.forcings.FT.data[:, :, 1] = (Q / cₚ) * (g.Az / (model.eos.ρ₀ * g.V))

    # # We will impose a surface heat flux of -400 W/m² ≈ -4.5e-6 K/s in a
    # square on the surface of the domain.
    #
    # As we can only impose forcing functions right now, and imposing a cooling
    # disk takes a pretty complicated function, I'll just impose a cooling square
    # until we can figure out the best way of imposing forcings with complex
    # geometries, probably by being able to define, e.g. a forcing only at the
    # top, etc.
    @inline function cooling_disk(u, v, w, T, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
        if k == 1
            x = i*Δx
            y = j*Δy
            Lx = Nx*Δx
            Ly = Ny*Δy
            r² = (x - Lx/2)^2 + (y - Ly/2)^2
            if r² < 600^2
                return -4.5e-6
            else
                return 0
            end
        else
            return 0
        end
    end

    model.forcing = Forcing(nothing, nothing, nothing, cooling_disk, nothing)
    nothing
end

Nx, Ny, Nz = 100, 100, 50
Lx, Ly, Lz = 2000, 2000, 1000
Nt, Δt = 1000, 20

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz))  # CPU Model
# model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), arch=:GPU, float_type=Float32)  # GPU Model

impose_cooling_disk!(model)

# Add a NetCDF output writer that saves NetCDF files to the current directory
# "." with a filename prefix of "deep_convection_" every 200 iterations.
nc_writer = NetCDFOutputWriter(dir=".", prefix="deep_convection_", frequency=20)
push!(model.output_writers, nc_writer)

# time_step!(model; Nt=Nt, Δt=Δt)
for i = 1:Nt
    tic = time_ns()
    time_step!(model, 1, Δt)
    println("Time: $(model.clock.time)  [$(prettytime(time_ns()-tic))]")
end

make_vertical_slice_movie(model, nc_writer, "T", Nt, Δt, 293.15, ceil(Int, Ny/2))
