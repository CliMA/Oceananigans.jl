TurbulentDiffusivities(arch::AbstractArchitecture, grid::AbstractGrid, tracers, ::AbstractLeith) =
    (νₑ=CellField(arch, grid),)

#####
##### The turbulence closure proposed by Leith
#####

"""
    TwoDimensionalLeith{FT} <: AbstractLeith{FT}

Parameters for the Smagorinsky-Lilly turbulence closure.
"""
struct TwoDimensionalLeith{FT, CR, GM} <: AbstractLeith{FT}
         C :: FT
    C_Redi :: CR
      C_GM :: GM
    function TwoDimensionalLeigth{FT}(C, C_Redi, C_GM)
        C_Redi = convert_diffusivity(FT, C_Redi)
        C_GM = convert_diffusivity(FT, C_GM)
        return new{FT, typeof(C_Redi), typeof(C_GM)}(C, C_Redi, C_GM)
    end
end

"""
    TwoDimensionalLeith([FT=Float64;] C=0.3, C_Redi=1, C_GM=1)

Return a `TwoDimensionalLeith` type associated with the turbulence closure proposed by
Leith (1965) and Kemper and Menemenlis (2008) which has an eddy viscosity of the form

    `νₑ = (C * Δᶠ)³ * √(ζ² + (∇h ∂z w)²)`

and an eddy diffusivity of the form...

where `Δᶠ` is the filter width, `ζ² = (∂x v - ∂y u)²` is the squared vertical vorticity,
and `C` is a model constant

Keyword arguments
=================
    - `C`      : Model constant
    - `C_Redi` : Coefficient for down-gradient tracer diffusivity for each tracer. i
                 Either a constant applied to every 
                 tracer, or a `NamedTuple` with fields for each tracer individually.
    - `C_GM`   : Coefficient for down-gradient tracer diffusivity for each tracer. i
                 Either a constant applied to every 
                 tracer, or a `NamedTuple` with fields for each tracer individually.

References
==========
Pearson, B. et al., "Evaluation of scale-aware subgrid mesoscale eddy models in a global eddy
rich model." Ocean Modelling (2017)
"""
TwoDimensionalLeith(FT=Float64; C=0.23, C_Redi=1, C_GM=1) = TwoDimensionalLeith{FT}(C, C_Redi, C_GM)

function with_tracers(tracers, closure::TwoDimensionalLeith{FT}) where FT
    C_Redi = tracer_diffusivities(tracers, closure.C_Redi)
    C_GM = tracer_diffusivities(tracers, closure.C_GM)
    return TwoDimensionalLeith{FT}(closure.C, C_Redi, C_GM)
end

function abs²_∇h_ζ(i, j, k, grid, U)
    vxx = ▶y_aca(i, j, k, grid, ∂x²_caa, U.v)
    uyy = ▶x_caa(i, j, k, grid, ∂y²_aca, U.u)
    uxy = ▶y_aca(i, j, k, grid, ∂x_caa, ∂y_afa, U.u)
    vxy = ▶x_caa(i, j, k, grid, ∂x_faa, ∂y_aca, U.v)

    return (vxx - uxy)^2 + (vxy - uyy)^2
end

@inline ψ²(i, j, k, grid, ψ, args...) = ψ(i, j, k, grid, args...)^2

function abs²_∇h_wz(i, j, k, grid, w)
    wxz² = ▶x_caa(i, j, k, grid, ϕ², ∂x_faa, ∂z_aac, w)
    wyz² = ▶y_aca(i, j, k, grid, ϕ², ∂y_afa, ∂z_aac, w)
    return wxz² + wyz²
end

@inline ν_ccc(i, j, k, grid, clo::TwoDimensionalLeith{FT}, buoyancy, U, C) where FT =
    (clo.C * Δᶠ(i, j, k, grid, clo))^3 * sqrt(  abs²_∇h_ζ(i, j, k, grid, C) 
                                              + abs²_∇h_wz(i, j, k, grid, U.w))

#####
##### Abstract Smagorinsky functionality
#####

# Components of the Redi rotation tensor

@inline function Redi_tensor_xz_fcc(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    bx = ∂x_b(i, j, k, grid, buoyancy, C) 
    bz = ▶xz_fac(i, j, k, grid, ∂z_b, buoyancy, C)
    return ifelse(bx == 0 && bz == 0, zero(FT), - bx / bz)
end

@inline function Redi_tensor_xz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    bx = ▶xz_caf(i, j, k, grid, ∂x_b, buoyancy, C)
    bz = ∂z_b(i, j, k, grid, buoyancy, C) 
    return ifelse(bx == 0 && bz == 0, zero(FT), - bx / bz)
end

@inline function Redi_tensor_yz_cfc(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    by = ∂y_b(i, j, k, grid, buoyancy, C) 
    bz = ▶yz_afc(i, j, k, grid, ∂z_b, buoyancy, C)
    return ifelse(by == 0 && bz == 0, zero(FT), - by / bz)
end

@inline function Redi_tensor_yz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    by = ▶yz_acf(i, j, k, grid, ∂y_b, buoyancy, C)
    bz = ∂z_b(i, j, k, grid, buoyancy, C) 
    return ifelse(by == 0 && bz == 0, zero(FT), - by / bz)
end

@inline function Redi_tensor_zz_ccf(i, j, k, grid::AbstractGrid{FT}, buoyancy, C) where FT
    bx = ▶xz_caf(i, j, k, grid, ∂x_b, buoyancy, C)
    by = ▶yz_acf(i, j, k, grid, ∂y_b, buoyancy, C)
    bz = ∂z_b(i, j, k, grid, buoyancy, C) 
    return ifelse(by == 0 && && bx == 0 && bz == 0, zero(FT), (bx^2 + by^2) / bz^2)
end

# Diffusive fluxes for Leith diffusivities

"""
    K₁ⱼ_∂ⱼ_c(i, j, k, grid, c, tracer, closure, νₑ)

Return `K₁₁ ∂x c + K₁₃ ∂z c` for a Leith diffusivity.
"""
@inline function K₁ⱼ_∂ⱼ_c(i, j, k, grid, closure::AbstractLeith, 
                          c, ::Val{tracer_index}, νₑ, C, buoyancy) where tracer_index

    @inbounds C_Redi = closure.C_Redi[tracer_index]
    @inbounds C_GM = closure.C_GM[tracer_index]

    νₑ = ▶x_faa(i, j, k, grid, νₑ, closure)

    ∂x_c = ∂x_faa(i, j, k, grid, c)
    ∂z_c = ▶xz_fac(i, j, k, grid, ∂z_aaf, c)

    R₁₃ = Redi_tensor_xz_fcc(i, j, k, grid, buoyancy, C)

    return νₑ * (                 C_Redi * ∂x_c 
                 + (C_Redi - C_GM) * R₁₃ * ∂z_c)
end

"""
    K₂ⱼ_∂ⱼ_c(i, j, k, grid, c, tracer, closure, νₑ)

Return `K₂₂ ∂y c + K₂₃ ∂z c` for a Leith diffusivity.
"""
@inline function K₂ⱼ_∂ⱼ_c(i, j, k, grid, closure::AbstractLeith, 
                          c, ::Val{tracer_index}, νₑ, C, buoyancy) where tracer_index

    @inbounds C_Redi = closure.C_Redi[tracer_index]
    @inbounds C_GM = closure.C_GM[tracer_index]

    νₑ = ▶y_afa(i, j, k, grid, νₑ, closure)

    ∂y_c = ∂y_afa(i, j, k, grid, c)
    ∂z_c = ▶yz_afc(i, j, k, grid, ∂z_aaf, c)

    R₂₃ = Redi_tensor_yz_cfc(i, j, k, grid, buoyancy, C)
    return νₑ * (                 C_Redi * ∂y_c 
                 + (C_Redi - C_GM) * R₂₃ * ∂z_c)
end

"""
    K₃ⱼ_∂ⱼ_c(i, j, k, grid, c, tracer, closure, νₑ)

Return `K₃₁ ∂x c + K₃₂ ∂y c + K₃₃ ∂z c` for a Leith diffusivity.
"""
@inline function K₃ⱼ_∂ⱼ_c(i, j, k, grid, closure::AbstractLeith, 
                          c, ::Val{tracer_index}, νₑ, C, buoyancy) where tracer_index

    @inbounds C_Redi = closure.C_Redi[tracer_index]
    @inbounds C_GM = closure.C_GM[tracer_index]

    νₑ = ▶z_aaf(i, j, k, grid, νₑ, closure)

    ∂x_c = ▶xz_caf(i, j, k, grid, ∂x_faa, c)
    ∂y_c = ▶yz_acf(i, j, k, grid, ∂y_afa, c)
    ∂z_c = ∂z_aaf(i, j, k, grid, c)

    R₃₁ = Redi_tensor_xz_ccf(i, j, k, grid, buoyancy, C)
    R₃₂ = Redi_tensor_yz_ccf(i, j, k, grid, buoyancy, C)
    R₃₃ = Redi_tensor_zz_ccf(i, j, k, grid, buoyancy, C)

    return νₑ * (
          (C_Redi + C_GM) * R₃₁ * ∂x_c 
        + (C_Redi + C_GM) * R₃₂ * ∂y_c 
                 + C_Redi * R₃₃ * ∂z_c)
end

"""
    ∇_κ_∇c(i, j, k, grid, c, closure, diffusivities)

Return the diffusive flux divergence `∇ ⋅ (κ ∇ c)` for the turbulence
`closure`, where `c` is an array of scalar data located at cell centers.
"""
@inline ∇_κ_∇c(i, j, k, grid, closure::AbstractLeith, c, tracer_index, 
               diffusivities, C, buoyancy) = (
      ∂x_caa(i, j, k, grid, K₁ⱼ_∂ⱼ_c, closure, c, tracer_index, diffusivities.νₑ, C, buoyancy)
    + ∂y_aca(i, j, k, grid, K₂ⱼ_∂ⱼ_c, closure, c, tracer_index, diffusivities.νₑ, C, buoyancy)
    + ∂z_aac(i, j, k, grid, K₃ⱼ_∂ⱼ_c, closure, c, tracer_index, diffusivities.νₑ, C, buoyancy)
)

function calculate_diffusivities!(K, arch, grid, closure::AbstractLeith, buoyancy, U, C)
    @launch(device(arch), config=launch_config(grid, 3), 
            calculate_nonlinear_viscosity!(K.νₑ, grid, closure, buoyancy, U, C))
end

"Return the filter width for a Leith Diffusivity on a Regular Cartesian grid."
@inline Δᶠ(i, j, k, grid::RegularCartesianGrid, ::AbstractLeith) = sqrt(grid.Δx * grid.Δy)
