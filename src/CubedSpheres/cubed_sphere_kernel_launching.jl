using KernelAbstractions: Event, MultiEvent

using Oceananigans.Architectures: device
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, PrescribedVelocityFields

using Oceananigans.Utils: launch!
import Oceananigans.Utils: launch!

maybe_replace_with_face(elem, cubed_sphere_grid, face_number) = elem

maybe_replace_with_face(t::Tuple, cubed_sphere_grid, face_number) = Tuple(maybe_replace_with_face(t_elem, cubed_sphere_grid, face_number) for t_elem in t)
maybe_replace_with_face(nt::NamedTuple, cubed_sphere_grid, face_number) = NamedTuple{keys(nt)}(maybe_replace_with_face(nt_elem, cubed_sphere_grid, face_number) for nt_elem in nt)

maybe_replace_with_face(grid::ConformalCubedSphereGrid, cubed_sphere_grid, face_number) = grid.faces[face_number]
maybe_replace_with_face(field::CubedSphereField, cubed_sphere_grid, face_number) = face(field, face_number)
maybe_replace_with_face(faces::CubedSphereFaces, cubed_sphere_grid, face_number) = faces[face_number]

maybe_replace_with_face(free_surface::ExplicitFreeSurface, cubed_sphere_grid, face_number) =
    ExplicitFreeSurface(face(free_surface.η, face_number), free_surface.gravitational_acceleration)

maybe_replace_with_face(velocities::PrescribedVelocityFields, cubed_sphere_grid, face_number) =
    PrescribedVelocityFields(face(velocities.u, face_number), face(velocities.v, face_number), face(velocities.w, face_number), velocities.parameters)

function launch!(arch, grid::ConformalCubedSphereGrid, args...; kwargs...)

    events = []
    for (face_number, grid_face) in enumerate(grid.faces)
        new_args = Tuple(maybe_replace_with_face(elem, grid, face_number) for elem in args)
        event = launch!(arch, grid_face, new_args...; kwargs...)
        push!(events, event)
    end

    # We should return the events but let's just wait here because errors.
    events = filter(e -> e isa Event, events)
    wait(device(arch), MultiEvent(Tuple(events)))

    # TODO: Other function expect an `event` to be returned.
    return events[1]
end
