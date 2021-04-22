using KernelAbstractions: Event, MultiEvent

using Oceananigans.Architectures: device
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, PrescribedVelocityFields

import Oceananigans.Utils: launch!

face(elem, cubed_sphere_grid, face_number) = elem

face(t::Tuple, cubed_sphere_grid, face_number) = Tuple(face(t_elem, cubed_sphere_grid, face_number) for t_elem in t)
face(nt::NamedTuple, cubed_sphere_grid, face_number) = NamedTuple{keys(nt)}(face(nt_elem, cubed_sphere_grid, face_number) for nt_elem in nt)

face(grid::ConformalCubedSphereGrid, cubed_sphere_grid, face_number) = grid.faces[face_number]
face(field::CubedSphereField, cubed_sphere_grid, face_number) = face(field, face_number)
face(faces::CubedSphereFaces, cubed_sphere_grid, face_number) = faces[face_number]

face(free_surface::ExplicitFreeSurface, cubed_sphere_grid, face_number) =
    ExplicitFreeSurface(face(free_surface.η, face_number), free_surface.gravitational_acceleration)

face(velocities::PrescribedVelocityFields, cubed_sphere_grid, face_number) =
    PrescribedVelocityFields(face(velocities.u, face_number), face(velocities.v, face_number), face(velocities.w, face_number), velocities.parameters)

function launch!(arch, grid::ConformalCubedSphereGrid, dims, kernel!, args...; kwargs...)

    events = []

    for (face_number, face_grid) in enumerate(grid.faces)
        face_args = Tuple(face(arg, grid, face_number) for arg in args)
        event = launch!(arch, face_grid, dims, kernel!, face_args...; kwargs...)
        push!(events, event)
    end

    events = filter(e -> e isa Event, events)

    return MultiEvent(Tuple(events))
end
