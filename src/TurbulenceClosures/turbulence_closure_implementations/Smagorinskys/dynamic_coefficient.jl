using Oceananigans.Architectures: architecture
using Oceananigans.Fields: interpolate
using Statistics

struct DynamicCoefficient{A, FT, S}
    averaging :: A
    minimum_numerator :: FT
    schedule :: S
end

const DynamicSmagorinsky = Smagorinsky{<:Any, <:DynamicCoefficient}

function DynamicSmagorinsky(time_discretization=ExplicitTimeDiscretization(), FT=Float64; averaging,
                            Pr=1.0, schedule=IterationInterval(1), minimum_numerator=1e-32)
    coefficient = DynamicCoefficient(FT; averaging, schedule, minimum_numerator)
    TD = typeof(time_discretization)
    Pr = convert_diffusivity(FT, Pr; discrete_form=false)
    return Smagorinsky{TD}(coefficient, Pr)
end

DynamicSmagorinsky(FT::DataType; kwargs...) = DynamicSmagorinsky(ExplicitTimeDiscretization(), FT; kwargs...)

Adapt.adapt_structure(to, dc::DynamicCoefficient) = DynamicCoefficient(dc.averaging, dc.minimum_numerator, nothing)

const DirectionallyAveragedCoefficient{N} = DynamicCoefficient{<:Union{NTuple{N, Int}, Int, Colon}} where N
const DirectionallyAveragedDynamicSmagorinsky{N} = Smagorinsky{<:Any, <:DirectionallyAveragedCoefficient{N}} where N

struct LagrangianAveraging end
const LagrangianAveragedCoefficient = DynamicCoefficient{<:LagrangianAveraging}
const LagrangianAveragedDynamicSmagorinsky = Smagorinsky{<:Any, <:LagrangianAveragedCoefficient}

"""
    DynamicCoefficient([FT=Float64;] averaging, schedule=IterationInterval(1), minimum_numerator=1e-32)

When used with `Smagorinsky`, it calculates the Smagorinsky coefficient dynamically from the flow
according to the scale invariant procedure in [BouZeid05](@citet).

`DynamicCoefficient` requires an `averaging` procedure, which can be a `LagrangianAveraging` (which
averages fluid parcels along their Lagrangian trajectory) or a tuple of integers indicating
a directional averaging procedure along chosen dimensions (e.g. `averaging=(1,2)` uses averages
in the `x` and `y` directions).

`DynamicCoefficient` is updated according to `schedule`, and `minimum_numerator` defines the minimum
value that is acceptable in the denominator of the final calculation.
"""
function DynamicCoefficient(FT=Float64; averaging, schedule=IterationInterval(1), minimum_numerator=1e-32)
    minimum_numerator = convert(FT, minimum_numerator)
    return DynamicCoefficient(averaging, minimum_numerator, schedule)
end

Base.summary(dc::DynamicCoefficient) = string("DynamicCoefficient(averaging = $(dc.averaging), schedule = $(dc.schedule))")
Base.show(io::IO, dc::DynamicCoefficient) = print(io, "DynamicCoefficient with\n",
                                                      "├── averaging = ", dc.averaging, "\n",
                                                      "├── schedule = ", dc.schedule, "\n",
                                                      "└── minimum_numerator = ", dc.minimum_numerator)

#####
##### Some common utilities independent of averaging
#####

@inline function square_smagorinsky_coefficient(i, j, k, grid, closure::DynamicSmagorinsky, diffusivity_fields, args...)
    𝒥ᴸᴹ = diffusivity_fields.𝒥ᴸᴹ
    𝒥ᴹᴹ = diffusivity_fields.𝒥ᴹᴹ
    𝒥ᴸᴹ_min = closure.coefficient.minimum_numerator

    @inbounds begin
        𝒥ᴸᴹ_ijk = max(𝒥ᴸᴹ[i, j, k], 𝒥ᴸᴹ_min)
        𝒥ᴹᴹ_ijk = 𝒥ᴹᴹ[i, j, k]
    end

    return ifelse(𝒥ᴹᴹ_ijk == 0, zero(grid), 𝒥ᴸᴹ_ijk / 𝒥ᴹᴹ_ijk)
end

@kernel function _compute_Σ_Σ̄!(Σ, Σ̄, grid, u, v, w)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Σ[i, j, k] = √(ΣᵢⱼΣᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
        Σ̄[i, j, k] = √(Σ̄ᵢⱼΣ̄ᵢⱼᶜᶜᶜ(i, j, k, grid, u, v, w))
    end
end

@kernel function _compute_LM_MM!(LM, MM, Σ, Σ̄, grid, u, v, w)
    i, j, k = @index(Global, NTuple)
    @info "                 Inside _compute_LM_MM!"
    @info "                 Calling LL_and_MM"
    LM_ijk, MM_ijk = LM_and_MM(i, j, k, grid, Σ, Σ̄, u, v, w)
    @info "                 Finished LM_and_MM"
    @inbounds begin
        LM[i, j, k] = LM_ijk
        MM[i, j, k] = MM_ijk
    end
end

@inline function LM_and_MM(i, j, k, grid, Σ, Σ̄, u, v, w)
    L₁₁ = L₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₂₂ = L₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₃₃ = L₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₁₂ = L₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₁₃ = L₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w)
    L₂₃ = L₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w)

    M₁₁ = M₁₁ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Σ, Σ̄)
    M₂₂ = M₂₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Σ, Σ̄)
    M₃₃ = M₃₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Σ, Σ̄)
    #M₁₂ = M₁₂ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Σ, Σ̄)
    #M₁₃ = M₁₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Σ, Σ̄)
    #M₂₃ = M₂₃ᶜᶜᶜ(i, j, k, grid, u, v, w, 2, 1, Σ, Σ̄)

    LM_ijk = L₁₁ * M₁₁ + L₂₂ * M₂₂ + L₃₃ * M₃₃ #+ 2L₁₂ * M₁₂ + 2L₁₃ * M₁₃ + 2L₂₃ * M₂₃
    MM_ijk = M₁₁ * M₁₁ + M₂₂ * M₂₂ + M₃₃ * M₃₃ #+ 2M₁₂ * M₁₂ + 2M₁₃ * M₁₃ + 2M₂₃ * M₂₃

    return LM_ijk, MM_ijk
end

#####
##### Directionally-averaged functionality
#####

function compute_coefficient_fields!(diffusivity_fields, closure::DirectionallyAveragedDynamicSmagorinsky, model; parameters)
    grid = model.grid
    arch = architecture(grid)
    velocities = model.velocities
    cˢ = closure.coefficient

    if cˢ.schedule(model)
        Σ = diffusivity_fields.Σ
        Σ̄ = diffusivity_fields.Σ̄
        launch!(arch, grid, :xyz, _compute_Σ_Σ̄!, Σ, Σ̄, grid, velocities...)

        LM = diffusivity_fields.LM
        MM = diffusivity_fields.MM
        launch!(arch, grid, :xyz, _compute_LM_MM!, LM, MM, Σ, Σ̄, grid, velocities...)

        𝒥ᴸᴹ = diffusivity_fields.𝒥ᴸᴹ
        𝒥ᴹᴹ = diffusivity_fields.𝒥ᴹᴹ
        compute!(𝒥ᴸᴹ)
        compute!(𝒥ᴹᴹ)
    end

    return nothing
end

function allocate_coefficient_fields(closure::DirectionallyAveragedDynamicSmagorinsky, grid)
    LM = CenterField(grid)
    MM = CenterField(grid)

    Σ = CenterField(grid)
    Σ̄ = CenterField(grid)

    𝒥ᴸᴹ = Field(Average(LM, dims=closure.coefficient.averaging))
    𝒥ᴹᴹ = Field(Average(MM, dims=closure.coefficient.averaging))

    return (; Σ, Σ̄, LM, MM, 𝒥ᴸᴹ, 𝒥ᴹᴹ)
end

#####
##### Lagrangian-averaged functionality
#####

const c = Center()

@kernel function _lagrangian_average_LM_MM!(𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, 𝒥ᴸᴹ_min, Σ, Σ̄, grid, Δt, u, v, w)
    i, j, k = @index(Global, NTuple)
    LM, MM = LM_and_MM(i, j, k, grid, Σ, Σ̄, u, v, w)
    FT = eltype(grid)

    @inbounds begin
        𝒥ᴸᴹ⁻ᵢⱼₖ = max(𝒥ᴸᴹ⁻[i, j, k], 𝒥ᴸᴹ_min)
        𝒥ᴹᴹ⁻ᵢⱼₖ = 𝒥ᴹᴹ⁻[i, j, k]

        # Compute time scale
        𝒥ᴸᴹ𝒥ᴹᴹ = 𝒥ᴸᴹ⁻ᵢⱼₖ * 𝒥ᴹᴹ⁻ᵢⱼₖ

        T⁻ = convert(FT, 1.5) * Δᶠ(i, j, k, grid) / ∜(∜(𝒥ᴸᴹ𝒥ᴹᴹ))
        τ = Δt / T⁻
        ϵ = τ / (1 + τ)
                        
        # Compute interpolation
        x = xnode(i, j, k, grid, c, c, c)
        y = ynode(i, j, k, grid, c, c, c)
        z = znode(i, j, k, grid, c, c, c)

        # Displacements
        δx = u[i, j, k] * Δt
        δy = v[i, j, k] * Δt
        δz = w[i, j, k] * Δt

        # Prevent displacements from getting too big?
        Δx = Δxᶜᶜᶜ(i, j, k, grid)
        Δy = Δyᶜᶜᶜ(i, j, k, grid)
        Δz = Δzᶜᶜᶜ(i, j, k, grid)

        δx = clamp(δx, -Δx, Δx)
        δy = clamp(δy, -Δy, Δy)
        δz = clamp(δz, -Δz, Δz)

        # Previous locations
        x⁻ = x - δx
        y⁻ = y - δy
        z⁻ = z - δz
        X⁻ = (x⁻, y⁻, z⁻)

        itp_𝒥ᴹᴹ⁻ = interpolate(X⁻, 𝒥ᴹᴹ⁻, (c, c, c), grid)
        itp_𝒥ᴸᴹ⁻ = interpolate(X⁻, 𝒥ᴸᴹ⁻, (c, c, c), grid)

        # Take time-step
        𝒥ᴹᴹ[i, j, k] = ϵ * MM + (1 - ϵ) * itp_𝒥ᴹᴹ⁻

        𝒥ᴸᴹ★ = ϵ * LM + (1 - ϵ) * max(itp_𝒥ᴸᴹ⁻, 𝒥ᴸᴹ_min)
        𝒥ᴸᴹ[i, j, k] = max(𝒥ᴸᴹ★, 𝒥ᴸᴹ_min)
    end
end

function compute_coefficient_fields!(diffusivity_fields, closure::LagrangianAveragedDynamicSmagorinsky, model; parameters)
    @info "               Inside compute_coefficient_fields!"
    grid = model.grid
    arch = architecture(grid)
    clock = model.clock
    cˢ = closure.coefficient
    t⁻ = diffusivity_fields.previous_compute_time
    u, v, w = model.velocities

    Δt = clock.time - t⁻[]
    t⁻[] = model.clock.time

    @info "               Preamble done"
    if cˢ.schedule(model)
        Σ = diffusivity_fields.Σ
        Σ̄ = diffusivity_fields.Σ̄
        @info "               Lauching _compute_Σ_Σ̄"
        launch!(arch, grid, :xyz, _compute_Σ_Σ̄!, Σ, Σ̄, grid, u, v, w)

        parent(diffusivity_fields.𝒥ᴸᴹ⁻) .= parent(diffusivity_fields.𝒥ᴸᴹ)
        parent(diffusivity_fields.𝒥ᴹᴹ⁻) .= parent(diffusivity_fields.𝒥ᴹᴹ)

        𝒥ᴸᴹ⁻ = diffusivity_fields.𝒥ᴸᴹ⁻
        𝒥ᴹᴹ⁻ = diffusivity_fields.𝒥ᴹᴹ⁻
        𝒥ᴸᴹ  = diffusivity_fields.𝒥ᴸᴹ
        𝒥ᴹᴹ  = diffusivity_fields.𝒥ᴹᴹ
        𝒥ᴸᴹ_min = cˢ.minimum_numerator

        if !isfinite(clock.last_Δt) || Δt == 0 # first time-step
            @info "               Launching _compute_LM_MM! at t=0"
            launch!(arch, grid, :xyz, _compute_LM_MM!, 𝒥ᴸᴹ, 𝒥ᴹᴹ, Σ, Σ̄, grid, u, v, w)
            @info "               Finished _compute_LM_MM!"
            parent(𝒥ᴸᴹ) .= max(mean(𝒥ᴸᴹ), 𝒥ᴸᴹ_min)
            parent(𝒥ᴹᴹ) .= mean(𝒥ᴹᴹ)
        else
            @info "               Lauching _compute_LM_MM!"
            launch!(arch, grid, :xyz,
                    _lagrangian_average_LM_MM!, 𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, 𝒥ᴸᴹ_min, Σ, Σ̄, grid, Δt, u, v, w)
            @info "               Finished _compute_LM_MM!"

        end
    end
    @info "               Calculations done"

    return nothing
end

function allocate_coefficient_fields(closure::LagrangianAveragedDynamicSmagorinsky, grid)
    𝒥ᴸᴹ⁻ = CenterField(grid)
    𝒥ᴹᴹ⁻ = CenterField(grid)

    𝒥ᴸᴹ = CenterField(grid)
    𝒥ᴹᴹ = CenterField(grid)

    Σ = CenterField(grid)
    Σ̄ = CenterField(grid)

    previous_compute_time = Ref(zero(grid))

    return (; Σ, Σ̄, 𝒥ᴸᴹ, 𝒥ᴹᴹ, 𝒥ᴸᴹ⁻, 𝒥ᴹᴹ⁻, previous_compute_time)
end


