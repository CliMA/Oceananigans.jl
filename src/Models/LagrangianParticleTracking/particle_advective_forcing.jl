using Adapt

@inline no_advective_forcing(args...) = nothing

struct ParticleAdvectiveForcing{U, V, W}
    u :: U
    v :: V
    w :: W
end

function ParticleAdvectiveForcing(; u=no_advective_forcing, v=no_advective_forcing, w=no_advective_forcing)
    return ParticleAdvectiveForcing(u, v, w)
end

# @inline (af::ParticleAdvectiveForcing)(i, j, k, grid, clock, model_fields) = 0

Base.summary(::ParticleAdvectiveForcing) = string("ParticleAdvectiveForcing")

function Base.show(io::IO, af::ParticleAdvectiveForcing)
    
    print(io, summary(af), ":", "\n")

    print(io, "├── u: ", prettysummary(af.u), "\n",
              "├── v: ", prettysummary(af.v), "\n",
              "└── w: ", prettysummary(af.w))
end

Adapt.adapt_structure(to, af::ParticleAdvectiveForcing) =
    ParticleAdvectiveForcing(adapt(to, af.u), adapt(to, af.v), adapt(to, af.w))

# # fallback
# @inline with_advective_forcing(forcing, total_velocities) = total_velocities

# @inline with_advective_forcing(forcing::AdvectiveForcing, total_velocities) = 
#     (u = SumOfArrays{2}(forcing.u, total_velocities.u),
#      v = SumOfArrays{2}(forcing.v, total_velocities.v),
#      w = SumOfArrays{2}(forcing.w, total_velocities.w))

# # Unwrap the tuple within MultipleForcings
# @inline with_advective_forcing(mf::MultipleForcings, total_velocities) =
#     with_advective_forcing(mf.forcings, total_velocities)

# # Recurse over forcing tuples
# @inline with_advective_forcing(forcing::Tuple, total_velocities) = 
#     @inbounds with_advective_forcing(forcing[2:end], with_advective_forcing(forcing[1], total_velocities))

# # Terminate recursion
# @inline with_advective_forcing(forcing::NTuple{1}, total_velocities) = 
#     @inbounds with_advective_forcing(forcing[1], total_velocities)
