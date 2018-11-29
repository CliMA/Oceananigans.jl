#
#
# Functions for eigenvector/eigenvalue direct factorization solution to Aϕ=f

# Code requires packages
# Pkg.add("LinearAlgebra")
# using LinearAlgebra
# Pkg.add("SparseArrays")
# using SparseArrays
# Pkg.add("Arpack")
# using Arpack
# 
# The solveLinearSystem() function is general for any A that has orthogonal eigenvectors.
# The complete set of eigenvectors comprise N^2 (where N=nx*ny*nz) values so this code
# is only useful for small debugging exercises.
#
# The solve_poisson_3d_mbc_eig() function forms an A operator with cyclic boundary conditions in X and Y
# and Neumann is Z. The function is configured to have the same arguments as the FFT solver solve_poisson_3d_mbc()
#
# The following illustrates code for using these functions
#
#  f=rand(7,9,5);Lx=100.;Ly=200.;Lz=10.; 
#  ϕ=solve_poisson_3d_mbc_eig(f, Lx, Ly, Lz)
#  println("   ϕ (dimensional)",ϕ)
#
function solveLinearSystem(A,f)
 # Solve Aϕ=f
 tol=1.e-12
    
 # Use eigen (from Julia package LinearAlgebra) - does not support sparse matrix format
 # E=eigen(A);
 # L=E.values;
 # V=E.vectors;
    
 # Alternate using eig from ARPACK (from Julia package Arpack) - works with sparse matrix format
 N=size(A)[1];
 Le=fill(0.,N)
 Ve=spzeros(N,N)
 lAR,vAR=eigs(A,nev=N,which=:LM,tol=0.0);
 neAR=size(lAR)[1]
 Le[1:neAR]=lAR;
 Ve[:,1:neAR]=vAR;
 L=Le;
 V=Ve;
 # End alternate using ARPACK
    
 # display(Matrix(V))
    
 # Get amplitudes, F, of eigenvectors that give f
 F=V'*f
 rL=map(x -> if (abs(x)>tol) 1.0/x;  else 0. ; end , L);
 # Get amplitudes, Φ, of eigenvectors that give ϕ
 Φ=F.*rL
 # Solve for ϕ given Φ
 ϕ=V*Φ
 # println("A ",A)
 println("A*ϕ ",A*ϕ)
 println("  f ",f)
 err=A*ϕ-f;
 println("err ",err)
 println("sum(err) ",sum(err))
 println("   ϕ",ϕ)
 return ϕ
end

function solve_poisson_3d_mbc_eig(f, Lx, Ly, Lz)
    # Solve Aϕ=f using slow and expensive eigenvector and eigenvalue
    # factorization. 
    nx, ny, nz = size(f)
    
    # == Form sparse A matrix from diagonals ==
    N=nx*ny*nz;
    dx=Lx/nx;dy=Ly/ny;dz=Lz/nz;
    rdx2=1/dx^2;rdy2=1/dy^2;rdz2=1/dz^2;
    # A matrix element values. mdi - interior main diag, mdze - z edge main diagonal, odx - x off diagonal, ody - y off diagonal, odz - z off diagonal
    mdi=-2*rdx2-2*rdy2-2*rdz2;
    mdze=-2*rdx2-2*rdy2-1*rdz2;
    odx=1*rdx2;
    ody=1*rdy2;
    odz=1*rdz2;
    
    md=vcat( fill(mdze,nx*ny),
         fill(mdi, N-2*nx*ny),
         fill(mdze,nx*ny)
       )
    A=spdiagm( -(N-1) =>fill(odx,1),
           -(N-nx)=>fill(ody,nx),
           -nx*ny =>fill(odz,(nz-1)*nx*ny),
           -nx    =>fill(ody,N-nx),
           -1     =>fill(odx,N-1),
           0      =>md,
           +1     =>fill(odx,N-1),
           +nx    =>fill(ody,N-nx),
           +nx*ny =>fill(odz,N-nx*ny),
           +(N-nx)=>fill(ody,nx),
           +(N-1) =>fill(odx,1)
         )
    
    # Create vector view of f
    fv=reshape(f,(N));

    # Normalize/regularize
    # Remove mean from f
    mnfv=sum(fv)./size(fv)
    fv=fv.-mnfv;
    # Scale elements of A and f 
    ma=reduce(max,(abs.(A)));
    A=A./ma;
    fv=fv./ma;
    mxfv=reduce(max,(abs.(fv)));
    fv=fv./mxfv;
    
    ϕ=solveLinearSystem(A,fv)
    # Unnormalize
    ϕ=ϕ.*mxfv;
    
end
