# Adapt an offset CuArray to work nicely with CUDA kernels.
Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(adapt(to, parent(x)), x.offsets)

# Need to adapt SubArray indices as well.
# See: https://github.com/JuliaGPU/Adapt.jl/issues/16
#Adapt.adapt_structure(to, A::SubArray{<:Any,<:Any,AT}) where {AT} =
#    SubArray(adapt(to, parent(A)), adapt.(Ref(to), parentindices(A)))

####
#### Convinient definitions
####

const second = 1.0
const minute = 60.0
const hour   = 60minute
const day    = 24hour

KiB, MiB, GiB, TiB = 1024.0 .^ (1:4)

####
#### Pretty printing
####

# Source: https://github.com/JuliaCI/BenchmarkTools.jl/blob/master/src/trials.jl
function prettytime(t)
    if t < 1e-6
        value, units = t * 1e9, "ns"
    elseif t < 1e-3
        value, units = t * 1e6, "μs"
    elseif t < 1
        value, units = t * 1e3, "ms"
    elseif t < 60
        value, units = t, "s"
    else
        value, units = t / 60, "min"
    end
    return @sprintf("%.3f", value) * " " * units
end


# Source: https://stackoverflow.com/a/1094933
function pretty_filesize(s, suffix="B")
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]
        abs(s) < 1024 && return @sprintf("%3.1f %s%s", s, unit, suffix)
        s /= 1024
    end
    return @sprintf("%.1f %s%s", s, "Yi", suffix)
end

####
#### Creating fields by dispatching on architecture
####

function Base.zeros(T, ::CPU, grid)
    # Starting and ending indices for the offset array.
    i1, i2 = 1 - grid.Hx, grid.Nx + grid.Hx
    j1, j2 = 1 - grid.Hy, grid.Ny + grid.Hy
    k1, k2 = 1 - grid.Hz, grid.Nz + grid.Hz

    underlying_data = zeros(T, grid.Tx, grid.Ty, grid.Tz)
    OffsetArray(underlying_data, i1:i2, j1:j2, k1:k2)
end

function Base.zeros(T, ::GPU, grid)
    # Starting and ending indices for the offset CuArray.
    i1, i2 = 1 - grid.Hx, grid.Nx + grid.Hx
    j1, j2 = 1 - grid.Hy, grid.Ny + grid.Hy
    k1, k2 = 1 - grid.Hz, grid.Nz + grid.Hz

    underlying_data = CuArray{T}(undef, grid.Tx, grid.Ty, grid.Tz)
    underlying_data .= 0  # Gotta do this otherwise you might end up with a few NaN values!
    OffsetArray(underlying_data, i1:i2, j1:j2, k1:k2)
end

Base.zeros(T, ::CPU, grid, Nx, Ny, Nz) = zeros(T, Nx, Ny, Nz)
Base.zeros(T, ::GPU, grid, Nx, Ny, Nz) = zeros(T, Nx, Ny, Nz) |> CuArray

# Default to type of Grid
Base.zeros(arch, grid::Grid{T}) where T = zeros(T, arch, grid)
Base.zeros(arch, grid::Grid{T}, Nx, Ny, Nz) where T = zeros(T, arch, grid, Nx, Ny, Nz)

####
#### Courant–Friedrichs–Lewy (CFL) condition number calculation
####

function cell_advection_timescale(u, v, w, grid)
    umax = maximum(abs, u)
    vmax = maximum(abs, v)
    wmax = maximum(abs, w)

    Δx = grid.Δx
    Δy = grid.Δy
    Δz = grid.Δz

    return min(Δx/umax, Δy/vmax, Δz/wmax)
end

cell_advection_timescale(model) =
    cell_advection_timescale(model.velocities.u.data.parent,
                             model.velocities.v.data.parent,
                             model.velocities.w.data.parent,
                             model.grid)

####
#### Adaptive time stepping
####

"""
    TimeStepWizard(cfl=0.1, max_change=2.0, min_change=0.5, max_Δt=Inf, kwargs...)

Instantiate a `TimeStepWizard`. On calling `update_Δt!(wizard, model)`,
the `TimeStepWizard` computes a time-step such that
`cfl = max(u/Δx, v/Δy, w/Δz) Δt`, where `max(u/Δx, v/Δy, w/Δz)` is the
maximum ratio between model velocity and along-velocity grid spacing
anywhere on the model grid. The new `Δt` is constrained to change by a
multiplicative factor no more than `max_change` or no less than
`min_change` from the previous `Δt`, and to be no greater in absolute
magnitude than `max_Δt`.
"""
Base.@kwdef mutable struct TimeStepWizard{T}
              cfl :: T = 0.1
    cfl_diffusion :: T = 2e-2
       max_change :: T = 2.0
       min_change :: T = 0.5
           max_Δt :: T = Inf
               Δt :: T = 0.01
end

"""
    update_Δt!(wizard, model)

Compute `wizard.Δt` given the velocities and diffusivities
of `model`, and the parameters of `wizard`.
"""
function update_Δt!(wizard, model)
    Δt = wizard.cfl * cell_advection_timescale(model)

    # Put the kibosh on if needed
    Δt = min(wizard.max_change * wizard.Δt, Δt)
    Δt = max(wizard.min_change * wizard.Δt, Δt)
    Δt = min(wizard.max_Δt, Δt)

    wizard.Δt = Δt

    return nothing
end

#####
##### Some utilities for tupling
#####

tupleit(t::Tuple) = t
tupleit(a::AbstractArray) = Tuple(a)
tupleit(nt) = tuple(nt)

parenttuple(obj) = Tuple(f.data.parent for f in obj)

####
#### Data tuples
####

@inline datatuple(obj::Nothing) = nothing
@inline datatuple(obj::AbstractArray) = obj
@inline datatuple(obj::Field) = obj.data
@inline datatuple(obj::NamedTuple) = NamedTuple{propertynames(obj)}(datatuple(o) for o in obj)
@inline datatuples(objs...) = (datatuple(obj) for obj in objs)

function getindex(t::NamedTuple, r::AbstractUnitRange{<:Real})
    n = length(r)
    n == 0 && return ()
    elems = Vector{eltype(t)}(undef, n)
    names = Vector{Symbol}(undef, n)
    o = first(r) - 1
    for i = 1:n
        elem = t[o + i]
        name = propertynames(t)[o + i]
        @inbounds elems[i] = elem
        @inbounds names[i] = name
    end
    NamedTuple{Tuple(names)}(Tuple(elems))
end

####
#### Dynamic launch configuration
####

function launch_config(grid, dims)
    return function (kernel)
        fun = kernel.fun
        config = launch_configuration(fun)

        # adapt the suggested config from 1D to the requested grid dimensions
        if dims == 3
            threads = floor(Int, cbrt(config.threads))
            blocks = ceil.(Int, [grid.Nx, grid.Ny, grid.Nz] ./ threads)
            threads = [threads, threads, threads]
        elseif dims == 2
            threads = floor(Int, sqrt(config.threads))
            blocks = ceil.(Int, [grid.Nx, grid.Ny] ./ threads)
            threads = [threads, threads]
        else
            error("unsupported launch configuration")
        end

        return (threads=Tuple(threads), blocks=Tuple(blocks))
    end
end
