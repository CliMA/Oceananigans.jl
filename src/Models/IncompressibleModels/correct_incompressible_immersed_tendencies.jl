using Oceananigans.Grids: xnode, ynode, znode, Face, Center, AbstractGrid
using ForwardDiff
using LinearAlgebra
using Oceananigans.Fields: interpolate

import Oceananigans.TimeSteppers: correct_immersed_tendencies!

"""
    correct_immersed_tendencies!(model)
    
Correct the tendency terms to implement no-slip boundary conditions on an immersed boundary
 without the contribution from the non-hydrostatic pressure. 
Makes velocity vanish within the immersed surface.
"""

# check if incompressible model (only model it works with for now!)
correct_immersed_tendencies!(model::IncompressibleModel) =
    correct_immersed_tendencies!(model, model.immersed_boundary)

# if no immersed boundary, do nothing (no cost)
correct_immersed_tendencies!(model, ::Nothing) = nothing

# otherwise, unpack the model
function correct_immersed_tendencies!(model, immersed_boundary)

    workgroup, worksize = work_layout(model.grid, :xyz)

    barrier = Event(device(model.architecture))

    correct_immersed_tendencies_kernel! = _correct_immersed_tendencies!(device(model.architecture), workgroup, worksize)
    
    # event we want to occur, evaluate using kernel function
    correct_tendencies_event =
        correct_immersed_tendencies_kernel!(model.grid,
                                            immersed_boundary,
                                            model.velocities,
                                            model.grid.Δx, 
                                            model.grid.Δy, model.grid.Δz,
                                            dependencies=barrier)
    # wait for these things to happen before continuing in calculations
    wait(device(model.architecture), correct_tendencies_event)

    return nothing
end

@kernel function _correct_immersed_tendencies!(grid::AbstractGrid{FT}, immersed, velocities, Δx, Δy, Δz) where FT
    
    i, j, k = @index(Global, NTuple)
    
    # (x,y,z) for u velocity grid
    x = xnode(Face, i, grid)
    y = ynode(Center, j, grid)
    z = znode(Center, k, grid)
    
    @inbounds begin
        immersed_update(i, j, k, x, y, z, immersed, grid, velocities, dirichZero, 1, max_neighbor)
    end
   
    # (x,y,z) for v velocity grid
    x = xnode(Center, i, grid)
    y = ynode(Face, j, grid)
    z = znode(Center, k, grid)
    
    @inbounds begin
	immersed_update(i, j, k, x, y, z, immersed, grid, velocities, dirichZero, 2, max_neighbor)
    end
    
    # (x,y,z) for w velocity grid
    x = xnode(Center, i, grid)
    y = ynode(Center, j, grid)
    z = znode(Face, k, grid)
    
    @inbounds begin
        immersed_update(i, j, k, x, y, z, immersed, grid, velocities, dirichZero, 3, max_neighbor)
    end
end

# function to finds the most positive distance of neighboring nodes
max_neighbor(x, y, z, Δx, Δy, Δz, immersed_distance)= max(immersed_distance([x+Δx y z]),
                    immersed_distance([x-Δx y z]), immersed_distance([x y+Δy z]),
                    immersed_distance([x y-Δy z]), immersed_distance([x y z+Δz]),
                    immersed_distance([x y z-Δz]),immersed_distance([x+2*Δx y z]),
                    immersed_distance([x-2*Δx y z]), immersed_distance([x y+2*Δy z]),
                    immersed_distance([x y-2*Δy z]), immersed_distance([x y z+2*Δz]),
                    immersed_distance([x y z-2*Δz]))  

# function to find the tangential plane to the point for rotation to tangential and normal
function projection_matrix(N)
    if abs(N[1]) ==1 # if normal vector is entirely in the x direction
        v1 = [0; -sign(N[1]); 1]  # we want v1 = [0,0,1]
        v1 = v1./norm(v1); # normalizing vector
        v2 = [0; 0; -1];
    elseif abs(N[2]) == 1 || N[3]==0 # if normal vector is entirely in the y direction or 0 z comp
        v1 = [sign(N[2]); -N[1]*sign(N[2])/N[2]; 0] # we want v1 = [1,0,0]
        v1 = v1./norm(v1); # normalizing vector
        v2 = cross(N,v1); # cross product to be orthonormal to both vectors
    else
        v1 = [sign(N[3]); 0; -N[1]*sign(N[3])/N[3]]; # else we want v1 = [1, 0 , ?]
        v1 = v1./norm(v1); # normalizing vector
        v2 = cross(N,v1); # cross product to be orthonormal to both vectors
    end
    transpose(hcat(v1,v2,N)) # creating a matrix of all 3 vectors
end 


# dirichlet bc Vb to find enforced pt vel u1 given the value at u2 and the sfc normal distances 
# assume boundary is at dist = 0 and fluid is negative d,so if reflected equidistant : d1 = -d2 
function interp_bc(u2, d2, Vb)
    u1 = 2*Vb-u2
end
        
# hard coding dirichlet zero bc for now
dirichZero(x) = 0.        
    
# function to find the value to enforce at a given immersed node         
function immersed_value(xvec, im_dist, grid, velocities, b_conds, needed_idx)
    # a function that finds the gradient of the distance function, ie. the sfc normal vectors
    # requires function have vector input, not what we want in the long run

    normalDist = x-> ForwardDiff.gradient(im_dist,x)   
    xF = xvec; # forced node is the x argument
    n = normalDist(xF); # sfc normal vector
    d2 = abs(im_dist(xF)) # distance to surface
    x₀ = xF + d2*n #closest point on boundary
    xI = x₀ + d2*n #reflected point over boundary
    
    #interpolated velocities withw trilinear interpolation
    uI = interpolate(velocities.u, xI[1], xI[2], xI[3])
    vI = interpolate(velocities.v, xI[1], xI[2], xI[3])
    wI = interpolate(velocities.w, xI[1], xI[2], xI[3])
    
    # matrix to rotate into tangential and normal
    matrix = projection_matrix(n);
    # new velocity vector: V_rot = [Vt1, Vt2, Vn]
    V_rot = matrix*[uI; vI; wI];
    
    # velocities at forced point from interpolation and BCs
    # need some kind of thing to separate out boundary conditions into what we have
    VF¹ = interp_bc(V_rot[1], d2, b_conds(xF));
    VF² = interp_bc(V_rot[2], d2, b_conds(xF));
    VFⁿ = interp_bc(V_rot[3], d2, b_conds(xF));
    
    # now we need to return to cartesian coordinate system
    VF = inv(matrix)*[VF¹;VF²;VFⁿ];
    
    # now we need to choose the one velocity out of these that we are enforcing at this location
    vel = VF[needed_idx]
end

# function to update a particular velocity
function immersed_update(i, j, k, x, y, z, im_dist, grid, velocities, b_conds, needed_idx, max_neighbor)
    # a function that takes in a location, checks if immersed node, and forces velocities accordingly
    spc = [x, y, z]
    if im_dist(spc) < 0 # solid node
        if max_neighbor(x, y, z, grid.Δx, grid.Δy, grid.Δz, im_dist) > 0 # fluid neighbor
            velI = immersed_value(spc, im_dist, grid, velocities, b_conds, needed_idx)
            velocities[needed_idx][i, j, k] = velI
        else
            velocities[needed_idx][i, j, k] = 0.    
        end
    end
 end
