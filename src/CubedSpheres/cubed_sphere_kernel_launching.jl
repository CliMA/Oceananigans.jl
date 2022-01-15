using KernelAbstractions: Event, MultiEvent

using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: device
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, PrescribedVelocityFields

import Oceananigans.Utils: launch!

get_face(obj, face_index) = obj
get_face(t::Tuple, face_index) = Tuple(get_face(t_elem, face_index) for t_elem in t)
get_face(nt::NamedTuple, face_index) = NamedTuple{keys(nt)}(get_face(nt_elem, face_index) for nt_elem in nt)

get_face(grid::ConformalCubedSphereGrid, face_index) = grid.faces[face_index]
get_face(faces::CubedSphereFaces, face_index) = faces[face_index]

get_face(free_surface::ExplicitFreeSurface, face_index) =
    ExplicitFreeSurface(get_face(free_surface.η, face_index), free_surface.gravitational_acceleration)

get_face(velocities::PrescribedVelocityFields, face_index) =
    PrescribedVelocityFields(get_face(velocities.u, face_index),
                             get_face(velocities.v, face_index),
                             get_face(velocities.w, face_index),
                             velocities.parameters)

function get_face(op::KernelFunctionOperation, face_index)
    LX, LY, LZ = location(op)
    computed_dependencies = get_face(op.computed_dependencies, face_index)
    parameters = get_face(op.parameters, face_index)
    face_grid = get_face(op.grid, face_index)
    return KernelFunctionOperation{LX, LY, LZ}(op.kernel_function,
                                               computed_dependencies,
                                               parameters,
                                               face_grid)
end

function launch!(arch, grid::ConformalCubedSphereGrid, dims, kernel!, args...; kwargs...)

    events = []

    for (face_index, face_grid) in enumerate(grid.faces)
        face_args = Tuple(get_face(arg, face_index) for arg in args)
        event = launch!(arch, face_grid, dims, kernel!, face_args...; kwargs...)
        push!(events, event)
    end

    events = filter(e -> e isa Event, events)

    return MultiEvent(Tuple(events))
end

@inline launch!(arch, grid::ConformalCubedSphereGrid, ::Val{dims}, args...; kwargs...) where dims = launch!(arch, grid, dims, args...; kwargs...)
