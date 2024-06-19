using Oceananigans.Fields: AbstractField

#####
##### The turbulence closure proposed by Leith
#####

struct TwoDimensionalLeith{FT, CR, GM, M} <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, ThreeDimensionalFormulation, 2}
                  C :: FT
             C_Redi :: CR
               C_GM :: GM
    isopycnal_model :: M

    function TwoDimensionalLeith{FT}(C, C_Redi, C_GM, isopycnal_model) where FT
        C_Redi = convert_diffusivity(FT, C_Redi)
        C_GM = convert_diffusivity(FT, C_GM)
        return new{FT, typeof(C_Redi), typeof(C_GM), typeof(isopycnal_model)}(C, C_Redi, C_GM)
    end
end

"""
    TwoDimensionalLeith(FT=Float64;
                        C=0.3, C_Redi=1, C_GM=1,
                        isopycnal_model=SmallSlopeIsopycnalTensor())

Return a `TwoDimensionalLeith` type associated with the turbulence closure proposed by
[leith1968diffusion](@citet) and [Fox-Kemper2008](@citet) which has an eddy viscosity of the form

```julia
νₑ = (C * Δᶠ)³ * √(|∇ₕ ζ|² + |∇ₕ ∂w/∂z|²)
```

and an eddy diffusivity of the form...

where `Δᶠ` is the filter width, `ζ = ∂v/∂x - ∂u/∂y` is the vertical vorticity,
and `C` is a model constant.

Keyword arguments
=================

  - `C`: Model constant
  - `C_Redi`: Coefficient for down-gradient tracer diffusivity for each tracer.
              Either a constant applied to every tracer, or a `NamedTuple` with fields
              for each tracer individually.
  - `C_GM`: Coefficient for down-gradient tracer diffusivity for each tracer.
            Either a constant applied to every tracer, or a `NamedTuple` with fields
            for each tracer individually.

References
==========

Leith, C. E. (1968). "Diffusion Approximation for Two‐Dimensional Turbulence", The Physics of
    Fluids 11, 671. doi: 10.1063/1.1691968

Fox‐Kemper, B., & D. Menemenlis (2008), "Can large eddy simulation techniques improve mesoscale rich
    ocean models?", in Ocean Modeling in an Eddying Regime, Geophys. Monogr. Ser., 177, pp. 319–337.
    doi: 10.1029/177GM19
"""
TwoDimensionalLeith(FT=Float64; C=0.3, C_Redi=1, C_GM=1, isopycnal_model=SmallSlopeIsopycnalTensor()) =
    TwoDimensionalLeith{FT}(C, C_Redi, C_GM, isopycnal_model)

function with_tracers(tracers, closure::TwoDimensionalLeith{FT}) where FT
    C_Redi = tracer_diffusivities(tracers, closure.C_Redi)
    C_GM = tracer_diffusivities(tracers, closure.C_GM)

    return TwoDimensionalLeith{FT}(closure.C, C_Redi, C_GM, closure.isopycnal_model)
end

@inline function abs²_∇h_ζ(i, j, k, grid, u, v)
    ζx = ℑyᵃᶜᵃ(i, j, k, grid, ∂xᶜᶠᶜ, ζ₃ᶠᶠᶜ, u, v)
    ζy = ℑxᶜᵃᵃ(i, j, k, grid, ∂yᶠᶜᶜ, ζ₃ᶠᶠᶜ, u, v)
    return ζx^2 + ζy^2
end

const ArrayOrField = Union{AbstractArray, AbstractField}

@inline ψ²(i, j, k, grid, ψ::Function, args...) = ψ(i, j, k, grid, args...)^2
@inline ψ²(i, j, k, grid, ψ::ArrayOrField, args...) = @inbounds ψ[i, j, k]^2

@inline function abs²_∇h_wz(i, j, k, grid, w)
    wxz = ℑxᶜᵃᵃ(i, j, k, grid, ∂xᶠᶜᶜ, ∂zᶜᶜᶜ, w)
    wyz = ℑyᵃᶜᵃ(i, j, k, grid, ∂yᶜᶠᶜ, ∂zᶜᶜᶜ, w)
    return wxz^2 + wyz^2
end

@kernel function _compute_leith_viscosity!(νₑ, grid, closure::TwoDimensionalLeith{FT}, buoyancy, velocities, tracers) where FT 
    i, j, k = @index(Global, NTuple)
    u, v, w = velocities
    prefactor = (closure.C * Δᶠ(i, j, k, grid, closure))^3 
    dynamic_ν = sqrt(abs²_∇h_ζ(i, j, k, grid, u, v) + abs²_∇h_wz(i, j, k, grid, w))

    @inbounds νₑ[i, j, k] = prefactor * dynamic_ν
end

function compute_diffusivities!(diffusivity_fields, closure::TwoDimensionalLeith, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    velocities = model.velocities
    tracers = model.tracers
    buoyancy = model.buoyancy

    launch!(arch, grid, parameters, _compute_leith_viscosity!,
            diffusivity_fields.νₑ, grid, closure, buoyancy, velocities, tracers)

    return nothing
end

"Return the filter width for a Leith Diffusivity on a general grid."
@inline Δᶠ(i, j, k, grid, ::TwoDimensionalLeith) = sqrt(Δxᶜᶜᶜ(i, j, k, grid) * Δyᶜᶜᶜ(i, j, k, grid)) 

function DiffusivityFields(grid, tracer_names, bcs, ::TwoDimensionalLeith)
    default_eddy_viscosity_bcs = (; νₑ = FieldBoundaryConditions(grid, (Center, Center, Center)))
    bcs = merge(default_eddy_viscosity_bcs, bcs)
    return (; νₑ=CenterField(grid, boundary_conditions=bcs.νₑ))
end

@inline viscosity(::TwoDimensionalLeith, K) = K.νₑ
@inline diffusivity(::TwoDimensionalLeith, K, ::Val{id}) where id = K.νₑ   

#####
##### Abstract Smagorinsky functionality
#####

# Diffusive fluxes for Leith diffusivities

@inline function diffusive_flux_x(i, j, k, grid, closure::TwoDimensionalLeith, diffusivities, 
                                  ::Val{tracer_index}, c, clock, fields, buoyancy) where tracer_index

    νₑ = diffusivities.νₑ

    C_Redi = closure.C_Redi[tracer_index]
    C_GM = closure.C_GM[tracer_index]

    νₑⁱʲᵏ = ℑxᶠᵃᵃ(i, j, k, grid, νₑ)

    ∂x_c = ∂xᶠᶜᶜ(i, j, k, grid, c)
    ∂z_c = ℑxzᶠᵃᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₁₃ = isopycnal_rotation_tensor_xz_fcc(i, j, k, grid, buoyancy, fields, closure.isopycnal_model)

    return - νₑⁱʲᵏ * (                 C_Redi * ∂x_c
                      + (C_Redi - C_GM) * R₁₃ * ∂z_c)
end

@inline function diffusive_flux_y(i, j, k, grid, closure::TwoDimensionalLeith, diffusivities,
                                  ::Val{tracer_index}, c, clock, fields, buoyancy) where tracer_index

    νₑ = diffusivities.νₑ

    C_Redi = closure.C_Redi[tracer_index]
    C_GM = closure.C_GM[tracer_index]

    νₑⁱʲᵏ = ℑyᵃᶠᵃ(i, j, k, grid, νₑ)

    ∂y_c = ∂yᶜᶠᶜ(i, j, k, grid, c)
    ∂z_c = ℑyzᵃᶠᶜ(i, j, k, grid, ∂zᶜᶜᶠ, c)

    R₂₃ = isopycnal_rotation_tensor_yz_cfc(i, j, k, grid, buoyancy, fields, closure.isopycnal_model)
    return - νₑⁱʲᵏ * (                  C_Redi * ∂y_c
                             + (C_Redi - C_GM) * R₂₃ * ∂z_c)
end

@inline function diffusive_flux_z(i, j, k, grid, closure::TwoDimensionalLeith, diffusivities, 
                                  ::Val{tracer_index}, c, clock, fields, buoyancy) where tracer_index

    νₑ = diffusivities.νₑ

    C_Redi = closure.C_Redi[tracer_index]
    C_GM = closure.C_GM[tracer_index]

    νₑⁱʲᵏ = ℑzᵃᵃᶠ(i, j, k, grid, νₑ)

    ∂x_c = ℑxzᶜᵃᶠ(i, j, k, grid, ∂xᶠᶜᶜ, c)
    ∂y_c = ℑyzᵃᶜᶠ(i, j, k, grid, ∂yᶜᶠᶜ, c)
    ∂z_c = ∂zᶜᶜᶠ(i, j, k, grid, c)

    R₃₁ = isopycnal_rotation_tensor_xz_ccf(i, j, k, grid, buoyancy, fields, closure.isopycnal_model)
    R₃₂ = isopycnal_rotation_tensor_yz_ccf(i, j, k, grid, buoyancy, fields, closure.isopycnal_model)
    R₃₃ = isopycnal_rotation_tensor_zz_ccf(i, j, k, grid, buoyancy, fields, closure.isopycnal_model)

    return - νₑⁱʲᵏ * (
          (C_Redi + C_GM) * R₃₁ * ∂x_c
        + (C_Redi + C_GM) * R₃₂ * ∂y_c
                 + C_Redi * R₃₃ * ∂z_c)
end
