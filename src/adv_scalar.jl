"""
    wenos5(x2,us,si,rho)

Computes the flux divergence of a scalar on a C-grid, in a given direction. Still 2D for now, but will be updated soon.

This weno5 scheme is applied to finite difference models written in conservation form.

To compute mass flux divergence, compute wenos5(x2,u_stag,ones(size(rho)),rho) 

Inputs:
x2: M x N+1 radius grid
us: M x N+1 staggered velocity
si: M x N scalar
rho0: M x N density

Outputs:
fdiv: M x N Local tendency due to horizontal flux divergence

References
==========
Durran (2008), "Numerical methods for geophysical fluid dynamics."
"""
function RK3(f, Φ, Δt)
    Φ⋆  = Φ + f(Φ)  * Δt/3
    Φ⋆⋆ = Φ + f(Φ⋆) * Δt/2
    return Φ + f(Φ⋆⋆) * Δt
end

function wenos5(x2,us,si,rho0)

# Optimal weights
C0 = 3/10
C1 = 3/5
C2 = 1/10

# Weights exponent
n = 2


        # First do advz
        smt = size(si,1)
        s = rho0.*si
    
    
    # First, define the summation operators for computing the "b"s:
    # A given "b", "b_k" becomes large when the solution is not smooth in the interval of
    # the stencil "k", thus diminishing the weight for that interval.   
    
    # b0 matrices
    D1 = sparse(1:smt,1:smt,ones(1,smt),smt,smt);
    D2 = sparse(1:smt-1,2:smt,-2*ones(1,smt-1),smt,smt);
    D3 = sparse(1:smt-2,3:smt,ones(1,smt-2),smt,smt);
    b01 = D1+D2+D3;
    
    D1 = sparse(1:smt,1:smt,3*ones(1,smt),smt,smt);
    D2 = sparse(1:smt-1,2:smt,-4*ones(1,smt-1),smt,smt);
    D3 = sparse(1:smt-2,3:smt,ones(1,smt-2),smt,smt);
    b02 = D1+D2+D3;
    
    # b1 matrices
    D1 = sparse(2:smt,1:smt-1,ones(1,smt-1),smt,smt);
    D2 = sparse(1:smt,1:smt,-2*ones(1,smt),smt,smt);
    D3 = sparse(1:smt-1,2:smt,ones(1,smt-1),smt,smt);
    b11 = D1+D2+D3;
    
    D1 = sparse(2:smt,1:smt-1,ones(1,smt-1),smt,smt);
    D2 = sparse(1:smt-1,2:smt,-ones(1,smt-1),smt,smt);
    b12 = D1+D2;
    
    # b2 matrices
    D1 = sparse(3:smt,1:smt-2,ones(1,smt-2),smt,smt);
    D2 = sparse(2:smt,1:smt-1,-2*ones(1,smt-1),smt,smt);
    D3 = sparse(1:smt,1:smt,ones(1,smt),smt,smt);
    b21 = D1+D2+D3;
    
    D1 = sparse(3:smt,1:smt-2,ones(1,smt-2),smt,smt);
    D2 = sparse(2:smt,1:smt-1,-4*ones(1,smt-1),smt,smt);
    D3 = sparse(1:smt,1:smt,3*ones(1,smt),smt,smt);
    b22 = D1+D2+D3;
    
    # b0
    b0 = 13/12.*(b01*s).^2 + 1/4.*(b02*s).^2;
    # b1
    b1 = 13/12.*(b11*s).^2 + 1/4.*(b12*s).^2;
    # b2
    b2 = 13/12.*(b21*s).^2 + 1/4.*(b22*s).^2;
    
    # Missing a smoothness reference computation! Taking a default value for now
    eps = 10^-6;

    # a0, a1, a2
    a0 = C0./(b0+eps).^n;
    a1 = C1./(b1+eps).^n;
    a2 = C2./(b2+eps).^n;
    
    # Compute the weights! These are defined at the interval i+1/2
    w0 = a0./(a0 + a1 + a2);
    w1 = a1./(a0 + a1 + a2);
    w2 = a2./(a0 + a1 + a2);
    
    # Compute the individual fluxes for each weight:
    # p0 matrices
    D1 = sparse(1:smt,1:smt,1/3*ones(1,smt),smt,smt);
    D2 = sparse(1:smt-1,2:smt,5/6*ones(1,smt-1),smt,smt);
    D3 = sparse(1:smt-2,3:smt,-1/6*ones(1,smt-2),smt,smt);
    p0_mat = D1+D2+D3;
    p0 = p0_mat*s;
    
    # p1 matrices
    D1 = sparse(2:smt,1:smt-1,-1/6*ones(1,smt-1),smt,smt);
    D2 = sparse(1:smt,1:smt,5/6*ones(1,smt),smt,smt);
    D3 = sparse(1:smt-1,2:smt,1/3*ones(1,smt-1),smt,smt);
    p1_mat = D1+D2+D3;
    p1 = p1_mat*s;
    
    # p2 matrices
    D1 = sparse(3:smt,1:smt-2,1/3*ones(1,smt-2),smt,smt);
    D2 = sparse(2:smt,1:smt-1,-7/6*ones(1,smt-1),smt,smt);
    D3 = sparse(1:smt,1:smt,11/6*ones(1,smt),smt,smt);
    p2_mat = D1+D2+D3;
    p2 = p2_mat*s;
    
    # Sum over the stencils multiplied by the weights
    F = p0.*w0 + p1.*w1 + p2.*w2;
    
    # Set last fluxes to 0, and add a row for i=0-1/2, with F=0
    F(end,:) = 0;
    F = [zeros(1,size(F,2)); F];
    
    

        Fu = F.*us;
        fdiv = (Fu(2:end,:) - Fu(1:end-1,:))./(x2(2:end,:)-x2(1:end-1,:)); 

    
    

end
