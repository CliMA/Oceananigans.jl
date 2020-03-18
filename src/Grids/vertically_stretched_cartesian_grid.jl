"""
    VerticallyStretchedCartesianGrid{FT, TX, TY, TZ, R, A} <: AbstractGrid{FT, TX, TY, TZ}

A Cartesian grid with with constant horizontal grid spacings `Δx` and `Δy`, and
non-uniform or stretched vertical grid spacing `Δz` between cell centers and cell faces,
topology `{TX, TY, TZ}`, and coordinate ranges of type `R` (where a range can be used) and
`A` (where an array is needed).
"""
struct VerticallyStretchedCartesianGrid{FT, TX, TY, TZ, R, A} <: AbstractGrid{FT, TX, TY, TZ}

    # Number of grid points in (x,y,z).
     Nx :: Int
     Ny :: Int
     Nz :: Int
 
    # Halo size in (x,y,z).
     Hx :: Int
     Hy :: Int
     Hz :: Int
 
    # Domain size [m].
     Lx :: FT
     Ly :: FT
     Lz :: FT

    # Grid spacing [m].
     Δx :: FT
     Δy :: FT
    ΔzF :: A
    ΔzC :: A

    # Range of coordinates at the centers of the cells.
     xC :: R
     yC :: R
     zC :: A
 
    # Range of grid coordinates at the faces of the cells.
    # Note: there are Nx+1 faces in the x-dimension, Ny+1 in the y, and Nz+1 in the z.
     xF :: R
     yF :: R
     zF :: A

end

"""
    VerticallyStretchedCartesianGrid(FT=Float64, arch=CPU();
                                     size, halo=(1, 1, 1), topology=(Periodic, Periodic, Bounded),
                                     length=nothing, x=nothing, y=nothing, z=nothing, zF=nothing)

"""
function VerticallyStretchedCartesianGrid(FT=Float64, arch=CPU(); size, zF, halo=(1, 1, 1),
                                          topology=(Periodic, Periodic, Bounded),
                                          length=nothing, x=nothing, y=nothing, z=nothing)
                                          

    # Hack that allows us to use `size` and `length` as keyword arguments but then also
    # use the `size` and `length` functions.
    sz, len = size, length
    length = Base.length

    TX, TY, TZ = validate_topology(topology)
    Lx, Ly, Lz, x, y, z = validate_grid_size_and_length(sz, len, halo, x, y, z)

    Nx, Ny, Nz = sz
    Hx, Hy, Hz = halo

    Δx = convert(FT, Lx / Nx)
    Δy = convert(FT, Ly / Ny)

    x₁, x₂ = convert.(FT, [x[1], x[2]])
    y₁, y₂ = convert.(FT, [y[1], y[2]])
    z₁, z₂ = convert.(FT, [z[1], z[2]])

    xF = range(x₁, x₂; length=Nx+1)
    yF = range(y₁, y₂; length=Ny+1)

    xC = range(x₁ + Δx/2, x₂ - Δx/2; length=Nx)
    yC = range(y₁ + Δy/2, y₂ - Δy/2; length=Ny)

    zF, zC, ΔzF, ΔzC = validate_and_generate_variable_grid_spacing(FT, zF, z₁, z₂, Nz, Hz)

    return VerticallyStretchedCartesianGrid{FT, typeof(TX), typeof(TY), typeof(TZ), typeof(xF), typeof(zF)}(
        Nx, Ny, Nz, Hx, Hy, Hz, Lx, Ly, Lz, Δx, Δy, ΔzF, ΔzC, xC, yC, zC, xF, yF, zF)
end

get_grid_spacing(z::Function, k) = z(k)
get_grid_spacing(z::AbstractVector{T}, k) where T = z[k]

function set_halo_zF!(zF, Nz, Hz)
    Δ_bottom = zF[2] - zF[1]
    for k = 0 : -1 : 1 - Hz
        zF[k] = zF[k+1] - Δ_bottom
    end

    Δ_top = zF[Nz+1] - zF[Nz]
    for k = Nz + 2 : Nz + 1 + Hz
        zF[k] = zF[k-1] + Δ_bottom
    end

    return nothing
end

function generate_variable_grid_spacings_from_zF(FT, zF_source, Nz, Hz)
    zF  = OffsetArray(zeros(FT, Nz + 1 + 2Hz), 1 - Hz : Nz + 1 + Hz)
    ΔzF = OffsetArray(zeros(FT, Nz + 2Hz),     1 - Hz : Nz + Hz) 

    zC  = OffsetArray(zeros(FT, Nz + 2Hz),     1 - Hz : Nz + Hz)
    ΔzC = OffsetArray(zeros(FT, Nz + 2Hz - 1), 2 - Hz : Nz + Hz) # index downshift

    for k in 1:Nz+1
        zF[k] = get_grid_spacing(zF_source, k)
    end

    set_halo_zF!(zF, Nz, Hz)

    for k in eachindex(zC)
        zC[k] = (zF[k] + zF[k+1]) / 2
        ΔzF[k] = zF[k+1] - zF[k]
    end

    for k in eachindex(ΔzC)
        ΔzC[k] = zC[k] - zC[k-1]
    end

    return zF, zC, ΔzF, ΔzC
end

function validate_and_generate_variable_grid_spacing(FT, zF_source, z₁, z₂, Nz, Hz)
    zF, zC, ΔzF, ΔzC = generate_variable_grid_spacings_from_zF(FT, zF_source, Nz, Hz)

    !isapprox(zF[1],   z₁) && throw(ArgumentError("Bottom face zF[1]=$(zF[1]) must equal bottom endpoint z₁=$z₁"))
    !isapprox(zF[Nz+1], z₂) && throw(ArgumentError("Top face zF[Nz+1]=$(zF[Nz+1]) must equal top endpoint z₂=$z₂"))

    return zF, zC, ΔzF, ΔzC
end
