using Oceananigans
using Oceananigans.Solvers
using SparseArrays, LinearAlgebra, PartitionedArrays
using IterativeSolvers

function test_fdm(parts)

  u(x) = x[1]+x[2]
  f(x) = zero(x[1])

  lx = 2.0
  ls = (lx,lx,lx)
  nx = 10
  ns = (nx,nx,nx)
  n = prod(ns)
  h = lx/(nx-1)
  points = [(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
  coeffs = [-6,1,1,1,1,1,1]/(h^2)
  stencil = [ (coeff,CartesianIndex(point)) for (coeff,point) in zip(coeffs,points) ]

  #lx = 2.0
  #ls = (lx,lx)
  #nx = 4
  #ns = (nx,nx)
  #n = prod(ns)
  #h = lx/(nx-1)
  #points = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]
  #coeffs = [-4,1,1,1,1]/(h^2)
  #stencil = [ (coeff,CartesianIndex(point)) for (coeff,point) in zip(coeffs,points) ]

  # Use a Cartesian partition if possible
  if ndims(parts) == length(ns)
    rows = PRange(parts,ns) 
  else
    rows = PRange(parts,n)
  end

  # We don't need the ghost layer for the rhs
  # So, it can be allocated right now.
  b = PVector{Float64}(undef,rows)

  # We don't need the ghost layer for the exact solution
  # So, it can be allocated right now.
  x̂ = similar(b)

  # Loop over (owned) rows, fill the coo-vectors, rhs, and the exact solution
  # In this case, we always touch local rows, but arbitrary cols.
  # Thus, row ids can be readily stored in local numbering so that we do not need to convert
  # them later.
  I,J,V = map_parts(rows.partition,b.values,x̂.values) do rows,b,x̂
    cis = CartesianIndices(ns)
    lis = LinearIndices(cis)
    I = Int[]
    J = Int[]
    V = Float64[]
    for lid in rows.oid_to_lid
      i = rows.lid_to_gid[lid]
      ci = cis[i]
      xi = (Tuple(ci) .- 1) .* h
      x̂[lid] = u(xi)
      boundary = any(s->(1==s||s==nx),Tuple(ci))
      if boundary
        push!(I,lid)
        push!(J,i)
        push!(V,one(eltype(V)))
        b[lid] = u(xi)
      else
        for (v,dcj) in stencil
          cj = ci + dcj
          j = lis[cj]
          push!(I,lid)
          push!(J,j)
          push!(V,-v)
        end
        b[lid] = f(xi)
      end
    end
    I,J,V
  end

  # TODO fill b and x̂ while add_gids is communicating values.

  # Build a PRange taking the owned ids in rows plus ghost ids from the touched cols
  cols = add_gids(rows,J)

  # Now we can convert J to local numbering, I is already in local numbering.
  to_lids!(J,cols)

  # Build the PSparseMatrix from the coo-vectors (in local numbering)
  # and the data distribution described by rows and cols.
  A = PSparseMatrix(I,J,V,rows,cols;ids=:local)

  # The initial guess needs the ghost layer (that why we take cols)
  # in other to perform the product A*x in the cg solver.
  # We also need to set the boundary values
  x0 = PVector(0.0,cols)
  map_parts(x0.values,x0.rows.partition) do x,rows
    for lid in rows.oid_to_lid
      cis = CartesianIndices(ns)
      i = rows.lid_to_gid[lid]
      ci = cis[i]
      xi = (Tuple(ci) .- 1) .* h
      boundary = any(s->(1==s||s==nx),Tuple(ci))
      if boundary
        x[lid] = u(xi)
      end
    end
  end

  # When this call returns, x has the correct answer only in the owned values.
  # The values at ghost ids can be recovered with exchange!(x)
  x = copy(x0)
  IterativeSolvers.cg!(x,A,b,verbose=i_am_main(parts))

  # This compares owned values, so we don't need to exchange!
  @test norm(x-x̂) < 1.0e-5

end
