import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings # To suppress divide-by-zero warnings safely

class CutCellOceanModel2D_NonUniform_Expanded:
    """
    An expanded conceptual 2D (x-z slice) ocean model demonstrating cut cells (hFac)
    on a non-uniform, structured grid. Includes T/S tracers, linear EOS,
    hydrostatic pressure, vertical velocity from continuity, basic momentum tendency
    (pressure gradient, viscosity, forcing), vertical advection/diffusion, and time stepping.

    NOTE: Momentum advection is omitted for simplicity. Vertical velocity is diagnosed
          from horizontal flow divergence (continuity). Uses upwind advection.
    """
    def __init__(self, x_face_coords, z_face_coords, bathy_func,
                 dt=60.0, Kh=1.0, Kv=1e-4, Ah=10.0, Av=1e-3,
                 rho0=1025.0, g=9.81, alpha=2e-4, beta=7.4e-4, T0=10.0, S0=35.0,
                 wind_stress_tau_x=0.0, bottom_drag_coeff=1e-3):
        """
        Initializes the model grid, geometry, parameters, and state variables.

        Args:
            x_face_coords (np.array): 1D array of x-coordinates of vertical cell faces.
            z_face_coords (np.array): 1D array of z-coordinates of horizontal cell faces
                                       (surface=0, increasing negative downwards).
            bathy_func (callable): Function h(x) returning water depth (positive value).
                                    Must be vectorized or handle array input.
            dt (float): Time step (seconds).
            Kh (float): Horizontal tracer diffusivity (m^2/s).
            Kv (float): Vertical tracer diffusivity (m^2/s).
            Ah (float): Horizontal momentum viscosity (m^2/s).
            Av (float): Vertical momentum viscosity (m^2/s).
            rho0 (float): Reference density (kg/m^3).
            g (float): Acceleration due to gravity (m/s^2).
            alpha (float): Thermal expansion coefficient (1/degC).
            beta (float): Haline contraction coefficient (1/PSU).
            T0, S0 (float): Reference temperature and salinity for EOS.
            wind_stress_tau_x (float or callable): Surface wind stress in x (N/m^2).
                                                   If callable, tau_x(x) is used.
            bottom_drag_coeff (float): Quadratic bottom drag coefficient (dimensionless).
        """
        print("Initializing CutCellOceanModel2D_NonUniform_Expanded...")
        # --- Grid Coordinates ---
        self.x_face = np.asarray(x_face_coords)
        self.z_face = np.asarray(z_face_coords)
        self.nx = len(self.x_face) - 1
        self.nz = len(self.z_face) - 1

        if self.nx <= 0 or self.nz <= 0:
            raise ValueError("Must have at least one cell in each dimension.")
        if not np.all(np.diff(self.x_face) > 0):
            raise ValueError("x_face_coords must be monotonically increasing.")
        if not np.all(np.diff(self.z_face) < 0):
             raise ValueError("z_face_coords must be monotonically decreasing.")

        # --- Grid Cell Centers and Dimensions ---
        self.x_center = (self.x_face[:-1] + self.x_face[1:]) / 2.0
        self.z_center = (self.z_face[:-1] + self.z_face[1:]) / 2.0
        self.dx = np.diff(self.x_face)      # Size nx array
        self.dz = -np.diff(self.z_face)     # Size nz array (positive thickness)
        self.dxf = np.diff(self.x_center)   # Distance between cell centers i and i+1
        self.dzf = -np.diff(self.z_center)  # Distance between cell centers k and k+1 (positive)

        # --- Bathymetry ---
        self.bathy_func = bathy_func
        self.bathy_depth_c = -self.bathy_func(self.x_center) # At cell center x
        self.bathy_depth_w = -self.bathy_func(self.x_face[:-1]) # At west face x
        self.bathy_depth_e = -self.bathy_func(self.x_face[1:])  # At east face x

        # --- Geometric Factors, Masks, Centroids ---
        self.cell_area = np.outer(self.dx, self.dz)
        self.hFacC = np.zeros((self.nx, self.nz)) # Volume (Area in 2D) fraction at Center
        self.hFacW = np.zeros((self.nx, self.nz)) # Area fraction at West face
        self.hFacS = np.zeros((self.nx, self.nz)) # Area fraction at South (Bottom in 2D) face
        self.wet_area = np.zeros((self.nx, self.nz)) # dx*dz*hFacC
        self.wet_face_area_x = np.zeros((self.nx, self.nz)) # dz*hFacW (at West face i)
        self.wet_face_area_z = np.zeros((self.nx, self.nz)) # dx*hFacS (at Bottom face k)
        self.maskC = np.zeros((self.nx, self.nz), dtype=int) # Cell Center
        self.maskW = np.zeros((self.nx, self.nz), dtype=int) # West Face i
        self.maskS = np.zeros((self.nx, self.nz), dtype=int) # South (Bottom) Face k
        self.centroid_x = np.full((self.nx, self.nz), np.nan)
        self.centroid_z = np.full((self.nx, self.nz), np.nan)
        self.dist_cen_x = np.full((self.nx, self.nz), np.nan) # Distance between centroids i and i-1
        self.dist_cen_z = np.full((self.nx, self.nz), np.nan) # Distance between centroids k and k+1

        self._calculate_geometry_and_centroids()
        self._calculate_centroid_distances() # Needs centroids first

        # --- Physical Parameters ---
        self.dt = dt
        self.Kh = Kh
        self.Kv = Kv
        self.Ah = Ah
        self.Av = Av
        self.rho0 = rho0
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.T0 = T0
        self.S0 = S0
        self.wind_stress_tau_x = wind_stress_tau_x
        self.bottom_drag_coeff = bottom_drag_coeff

        # --- State Variables ---
        # Tracers at cell centers (centroids)
        self.T = np.full((self.nx, self.nz), self.T0) * self.maskC
        self.S = np.full((self.nx, self.nz), self.S0) * self.maskC
        # Velocity on faces
        self.u = np.zeros((self.nx, self.nz)) # U on west face i
        self.w = np.zeros((self.nx, self.nz)) # W on bottom face k (diagnosed)
        # Auxiliary fields at centers
        self.pressure = np.zeros((self.nx, self.nz)) # Hydrostatic pressure anomaly (relative to surface)
        self.density = np.full((self.nx, self.nz), self.rho0) * self.maskC

        # Tendencies (storage for updates)
        self.T_tendency = np.zeros((self.nx, self.nz))
        self.S_tendency = np.zeros((self.nx, self.nz))
        self.u_tendency = np.zeros((self.nx, self.nz))

        print("Initialization complete.")
        print(f"Grid: {self.nx}x{self.nz}. dt={self.dt}s.")
        print(f"Kh={self.Kh:.2e}, Kv={self.Kv:.2e}, Ah={self.Ah:.2e}, Av={self.Av:.2e}")

    def _calculate_geometry_and_centroids(self):
        # (This method is largely unchanged from the original, but ensure refinement happens at the end)
        print("Calculating geometric factors (hFac) and centroids...")
        full_face_area_x = self.dz # West/East face area depends only on k -> shape (nz,)
        full_face_area_z = self.dx # Bottom/Top face area depends only on i -> shape (nx,)

        for i in range(self.nx):
            for k in range(self.nz):
                # Cell boundaries
                z_top_face = self.z_face[k]
                z_bot_face = self.z_face[k+1]
                x_west_face = self.x_face[i]
                x_east_face = self.x_face[i+1]
                dz_k = self.dz[k]
                dx_i = self.dx[i]
                full_area_ik = self.cell_area[i, k]

                bathy_c = self.bathy_depth_c[i]
                bathy_w = self.bathy_depth_w[i]
                bathy_e = self.bathy_depth_e[i]

                # hFacC and maskC
                if z_top_face < bathy_c: hFacC = 0.0
                elif z_bot_face >= bathy_c: hFacC = 1.0
                else: hFacC = np.clip((bathy_c - z_bot_face) / dz_k, 0.0, 1.0)
                self.hFacC[i, k] = hFacC
                self.maskC[i, k] = 1 if hFacC > 1e-10 else 0
                self.wet_area[i, k] = full_area_ik * hFacC

                # hFacW and maskW (Initial estimate based on bathy_w)
                if z_top_face < bathy_w: hFacW = 0.0
                elif z_bot_face >= bathy_w: hFacW = 1.0
                else: hFacW = np.clip((bathy_w - z_bot_face) / dz_k, 0.0, 1.0)
                self.hFacW[i, k] = hFacW
                self.maskW[i, k] = 1 if hFacW > 1e-10 else 0
                self.wet_face_area_x[i, k] = full_face_area_x[k] * hFacW

                # hFacS and maskS (Initial estimate based on bathy_c)
                # Bottom face is open if the face itself is below the water at that location
                # Use bathy_c as approximation for bathy depth over the face
                bathy_at_z_bot = bathy_c
                if z_bot_face < bathy_at_z_bot: hFacS = 1.0
                else: hFacS = 0.0
                self.hFacS[i, k] = hFacS
                self.maskS[i, k] = 1 if hFacS > 1e-10 else 0
                self.wet_face_area_z[i, k] = full_face_area_z[i] * hFacS

                # --- Calculate Centroid (cx, cz) ---
                if self.maskC[i, k] == 0:
                    self.centroid_x[i, k], self.centroid_z[i, k] = np.nan, np.nan
                    continue

                z_bathy_at_west = np.clip(bathy_w, z_bot_face, z_top_face)
                z_bathy_at_east = np.clip(bathy_e, z_bot_face, z_top_face)
                h_wet_west = np.clip(z_bathy_at_west - z_bot_face, 0.0, dz_k)
                h_wet_east = np.clip(z_bathy_at_east - z_bot_face, 0.0, dz_k)

                if hFacC >= 1.0 - 1e-10: # Fully wet
                    cx = self.x_center[i]
                    cz = self.z_center[k]
                elif hFacC > 1e-10: # Partially wet
                    trapezoid_area = (h_wet_west + h_wet_east) / 2.0 * dx_i
                    if trapezoid_area < 1e-12 or abs(h_wet_west + h_wet_east) < 1e-12:
                        cx = self.x_center[i]
                        cz = z_bot_face + hFacC * dz_k * 0.5 # Approx centroid of partial cell
                    else:
                        cx = x_west_face + dx_i * (h_wet_east + 2 * h_wet_west) / (3 * (h_wet_west + h_wet_east))
                        cz = z_bot_face + (h_wet_west**2 + h_wet_west * h_wet_east + h_wet_east**2) / (3 * (h_wet_west + h_wet_east))

                    cx = np.clip(cx, x_west_face + 1e-9, x_east_face - 1e-9) # Ensure within bounds
                    cz = np.clip(cz, z_bot_face + 1e-9, z_top_face - 1e-9)
                else: # Effectively dry
                    cx, cz = np.nan, np.nan

                self.centroid_x[i, k] = cx
                self.centroid_z[i, k] = cz

        # --- Refine face masks based on adjacent *wet cells* ---
        print("Refining face masks based on connectivity...")
        # West face maskW[i,k] needs C[i,k] AND C[i-1,k] to be wet.
        self.maskW[0, :] = 0 # Domain boundary
        self.wet_face_area_x[0, :] = 0.0
        for i in range(1, self.nx):
            connectivity_mask = self.maskC[i, :] * self.maskC[i-1, :]
            self.maskW[i, :] = self.maskW[i, :] * connectivity_mask # Combine geometric and connectivity
            self.wet_face_area_x[i, self.maskW[i, :] == 0] = 0.0 # Zero area if not connected or geometrically blocked

        # Bottom face maskS[i,k] needs C[i,k] AND C[i,k+1] to be wet.
        self.maskS[:, -1] = 0 # Domain boundary (bottom)
        self.wet_face_area_z[:, -1] = 0.0
        for k in range(self.nz - 1):
            connectivity_mask = self.maskC[:, k] * self.maskC[:, k+1]
            self.maskS[:, k] = self.maskS[:, k] * connectivity_mask # Combine geometric and connectivity
            self.wet_face_area_z[self.maskS[:, k] == 0, k] = 0.0 # Zero area if not connected or geometrically blocked

        print("Geometric factors and centroid calculation complete.")

    def _calculate_centroid_distances(self):
        """Calculate distances between adjacent cell centroids."""
        print("Calculating centroid distances...")
        # Distance between centroid(i,k) and centroid(i-1,k)
        self.dist_cen_x[1:, :] = self.centroid_x[1:, :] - self.centroid_x[:-1, :]
        self.dist_cen_x[self.dist_cen_x <= 1e-10] = np.nan # Avoid zero/small distance

        # Distance between centroid(i,k) and centroid(i,k+1) (negative z-coord difference)
        self.dist_cen_z[:, :-1] = self.centroid_z[:, :-1] - self.centroid_z[:, 1:]
        self.dist_cen_z[self.dist_cen_z <= 1e-10] = np.nan # Avoid zero/small distance
        print("Centroid distance calculation complete.")


    def get_recip_wet_area(self):
        """ Safely calculates 1.0 / wet_area, returning 0 where wet_area is 0. """
        recip = np.zeros_like(self.wet_area)
        wet = self.wet_area > 1e-15
        recip[wet] = 1.0 / self.wet_area[wet]
        return recip

    def _linear_eos(self, T, S):
        """ Simple linear equation of state. """
        return self.rho0 * (1.0 - self.alpha * (T - self.T0) + self.beta * (S - self.S0))

    def update_density_pressure(self):
        """ Updates density from T/S and calculates hydrostatic pressure anomaly. """
        self.density = self._linear_eos(self.T, self.S) * self.maskC

        # Integrate hydrostatic pressure downwards
        # p(k) = p(k-1) + rho(k-1)*g*dz(k-1) (approx)
        # More accurately: use average density between levels or integrate rho*g dz
        # Simplified approach: Pressure at center k depends on density of cells above
        self.pressure[:, :] = 0.0 # Start with zero at the surface reference
        for k in range(1, self.nz):
             # Pressure difference across layer k-1 (top face k to top face k-1)
             # Use density of the layer above (k-1) and thickness dz[k-1]
             # Apply only where both cell k and k-1 are wet for integration path
             mask_k_and_km1 = self.maskC[:, k] * self.maskC[:, k-1]
             # Average density (simple) or density at k-1? Let's use density at k-1
             rho_eff = self.density[:, k-1]
             # Effective thickness: dz[k-1]? Centroid distance? Use dz[k-1] for simplicity.
             delta_p = rho_eff * self.g * self.dz[k-1]
             self.pressure[:, k] = self.pressure[:, k-1] + delta_p * mask_k_and_km1

        self.pressure *= self.maskC # Ensure dry cells have no pressure anomaly

    def diagnose_vertical_velocity(self):
        """Calculates vertical velocity W on bottom face 'k' to ensure volume conservation."""
        # div(Flux) = d(u*dz)/dx + d(w*dx)/dz = 0
        # Integral form: Sum(fluxes) = 0
        # Flux_East - Flux_West + Flux_North - Flux_South = 0
        # (u_E * Area_E) - (u_W * Area_W) + (w_N * Area_N) - (w_S * Area_S) = 0
        # Here, N is top face (k-1), S is bottom face (k). Area_N is wet_face_area_z[i, k-1], Area_S is wet_face_area_z[i, k].
        # We want w_S = w[i, k] (velocity on bottom face k).
        # w_S * Area_S = w_N * Area_N + (u_E * Area_E) - (u_W * Area_W)
        # Note: u_E is u[i+1, k], Area_E is wet_face_area_x[i+1, k]
        #       u_W is u[i, k],   Area_W is wet_face_area_x[i, k]
        #       w_N is w[i, k-1]

        self.w[:, :] = 0.0 # Initialize
        recip_wet_area_z = np.zeros_like(self.wet_face_area_z)
        valid_z = self.wet_face_area_z > 1e-15
        recip_wet_area_z[valid_z] = 1.0 / self.wet_face_area_z[valid_z]

        # Integrate divergence downwards from surface (k=0)
        # Assume w=0 at the effective surface (top face of k=0 cell) -> w[i, -1] = 0 conceptually
        flux_horiz_div = np.zeros((self.nx, self.nz))

        # Calculate horizontal flux divergence within each layer k
        flux_W = self.u * self.wet_face_area_x # Flux across west face i
        flux_E = np.zeros_like(flux_W)
        flux_E[:-1, :] = flux_W[1:, :] # Flux across east face i is west face i+1
        flux_horiz_div = flux_E - flux_W # Net horizontal flux *out* of cell (i,k)

        # Integrate vertically: w_flux_S(k) = w_flux_N(k) + horiz_div(k)
        w_flux_N = np.zeros(self.nx) # Flux across top face of current layer (starts at 0 for k=0)
        for k in range(self.nz):
            w_flux_S = w_flux_N + flux_horiz_div[:, k] # Total flux needing to exit bottom face k
            # Convert flux to velocity: w = Flux / Area
            self.w[:, k] = w_flux_S * recip_wet_area_z[:, k] * self.maskS[:, k] # Apply mask for bottom face
            # Flux leaving bottom face k becomes flux entering top face k+1
            w_flux_N = w_flux_S * self.maskS[:, k] # Pass only through open faces

        # Ensure w is zero where bottom face is masked
        self.w *= self.maskS

    def _calculate_tracer_flux_divergence(self, tracer):
        """ Calculates advective + diffusive flux divergence for a tracer field. """
        flux_div_adv = np.zeros_like(tracer)
        flux_div_diff = np.zeros_like(tracer)

        # --- Advection (Upwind) ---
        # Horizontal Fluxes (East - West)
        flux_adv_W = np.zeros_like(tracer)
        flux_adv_E = np.zeros_like(tracer)
        for i in range(1, self.nx): # Loop over west faces
            u_face = self.u[i, :]
            area_w = self.wet_face_area_x[i, :]
            mask_w = self.maskW[i, :]
            tracer_upstream = np.where(u_face > 0, tracer[i-1, :], tracer[i, :])
            flux_adv_W[i, :] = u_face * tracer_upstream * area_w * mask_w
        flux_adv_E[:-1, :] = flux_adv_W[1:, :] # East face i = West face i+1
        flux_div_adv += (flux_adv_E - flux_adv_W)

        # Vertical Fluxes (North - South)
        flux_adv_S = np.zeros_like(tracer) # Flux across South (bottom) face k
        flux_adv_N = np.zeros_like(tracer) # Flux across North (top) face k
        for k in range(self.nz - 1): # Loop over bottom faces (except very bottom)
             w_face = self.w[:, k] # w on bottom face k
             area_s = self.wet_face_area_z[:, k]
             mask_s = self.maskS[:, k]
             # Upwind: If w>0 (upward), use tracer from k+1. If w<=0 (downward), use tracer from k.
             tracer_upstream = np.where(w_face > 0, tracer[:, k+1], tracer[:, k])
             flux_adv_S[:, k] = w_face * tracer_upstream * area_s * mask_s
        flux_adv_N[:, 1:] = flux_adv_S[:, :-1] # North face k = South face k-1
        flux_div_adv += (flux_adv_N - flux_adv_S) # Add vertical divergence

        # --- Diffusion ---
        # Horizontal Diffusion (East - West)
        flux_diff_W = np.zeros_like(tracer)
        flux_diff_E = np.zeros_like(tracer)
        with warnings.catch_warnings(): # Suppress divide by nan warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            grad_x = (tracer[1:, :] - tracer[:-1, :]) / self.dist_cen_x[1:, :] # Gradient between i and i-1
        grad_x[np.isnan(grad_x)] = 0.0 # Set gradient to 0 if distance was invalid
        for i in range(1, self.nx): # Loop over west faces
            area_w = self.wet_face_area_x[i, :]
            mask_w = self.maskW[i, :]
            flux_diff_W[i, :] = -self.Kh * grad_x[i-1, :] * area_w * mask_w # Use grad between i-1 and i for face i
        flux_diff_E[:-1, :] = flux_diff_W[1:, :]
        flux_div_diff += (flux_diff_E - flux_diff_W)

        # Vertical Diffusion (North - South)
        flux_diff_S = np.zeros_like(tracer) # Flux across South (bottom) face k
        flux_diff_N = np.zeros_like(tracer) # Flux across North (top) face k
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Gradient between k and k+1 (careful with sign convention for dz_c)
            grad_z = (tracer[:, 1:] - tracer[:, :-1]) / (-self.dist_cen_z[:, :-1]) # dz_c was positive distance
        grad_z[np.isnan(grad_z)] = 0.0
        for k in range(self.nz - 1): # Loop over bottom faces
            area_s = self.wet_face_area_z[:, k]
            mask_s = self.maskS[:, k]
            flux_diff_S[:, k] = -self.Kv * grad_z[:, k] * area_s * mask_s # Use grad between k and k+1 for face k
        flux_diff_N[:, 1:] = flux_diff_S[:, :-1]
        flux_div_diff += (flux_diff_N - flux_diff_S)

        total_flux_div = flux_div_adv + flux_div_diff
        return total_flux_div * self.maskC # Return divergence only for wet cells

    def calculate_tracer_tendency(self):
        """ Calculates tendencies for T and S. """
        recip_wet_area = self.get_recip_wet_area()
        div_T = self._calculate_tracer_flux_divergence(self.T)
        div_S = self._calculate_tracer_flux_divergence(self.S)
        self.T_tendency = -div_T * recip_wet_area
        self.S_tendency = -div_S * recip_wet_area


    def _calculate_pressure_gradient_force(self):
        """ Calculates pressure gradient force term for u-momentum. (-1/rho0 * dp/dx) """
        pgf = np.zeros_like(self.u)
        # Calculate pressure difference across the u-face (west face i)
        # Uses pressure at centers i and i-1
        # Need distance between centers i and i-1 --> use self.dxf (approx) or dist_cen_x? Use dist_cen_x.
        with warnings.catch_warnings(): # Suppress divide by nan warning
             warnings.simplefilter("ignore", category=RuntimeWarning)
             dp_dx = (self.pressure[1:, :] - self.pressure[:-1, :]) / self.dist_cen_x[1:,:]
        dp_dx[np.isnan(dp_dx)] = 0.0

        pgf[1:, :] = -(1.0 / self.rho0) * dp_dx
        return pgf * self.maskW # Apply only at open west faces

    def _calculate_viscosity_force(self):
        """ Calculates horizontal and vertical viscosity terms for u-momentum. """
        visc_force = np.zeros_like(self.u)

        # Horizontal Viscosity: d/dx (Ah * du/dx) at the u-point (west face i)
        # Needs gradient of u at cell centers, then take divergence at u-point.
        # Simpler: Calculate Laplacian at u-point using adjacent u-points.
        # d/dx(du/dx) approx (u[i+1,k] - 2*u[i,k] + u[i-1,k]) / dx_u^2
        # dx_u is distance between u-points = dx[i] roughly? Use dx array.
        laplacian_h = np.zeros_like(self.u)
        dx_u_sq = np.zeros_like(self.u)

        dx_avg = (self.dx[1:-1] + self.dx[:-2]) / 2.0
        # Assign to the corresponding slice i=1..nx-2

        dx_u_sq[1:-1, :] = dx_avg[:, np.newaxis]**2 # Shape (38, 1)**2 -> broadcasts to (38, 20)

        # Avg dx centered at u point i
        
        dx_u_sq[dx_u_sq < 1e-15] = np.inf # Avoid division by zero

        laplacian_h[1:-1, :] = (self.u[2:, :] - 2 * self.u[1:-1, :] + self.u[:-2, :]) / dx_u_sq[1:-1,:]
        visc_force += self.Ah * laplacian_h

        # Vertical Viscosity: d/dz (Av * du/dz) at the u-point (west face i)
        # Needs u gradients on horizontal faces above/below u[i,k].
        # Approx: (u[i,k-1] - 2*u[i,k] + u[i,k+1]) / dz_w^2
        # dz_w is distance between W-faces = dz[k] roughly? Use dz array.
        laplacian_v = np.zeros_like(self.u)
        dz_w_sq = self.dz**2 # Thickness squared at level k
        dz_w_sq[dz_w_sq < 1e-10] = np.inf

        laplacian_v[:, 1:-1] = (self.u[:, :-2] - 2 * self.u[:, 1:-1] + self.u[:, 2:]) / dz_w_sq[1:-1]
        # Handle boundaries (assume no shear gradient = zero flux)
        # Top boundary (k=0): gradient = (u[i,1]-u[i,0])/dz[0] -> flux = Av*grad. Need second derivative...
        # Simplified Neumann zero gradient: Assume effective u[i,-1] = u[i,0]
        laplacian_v[:, 0] = (self.u[:, 1] - self.u[:, 0]) / dz_w_sq[0] # One-sided difference approx
        # Bottom boundary (k=nz-1): Assume effective u[i,nz] = u[i,nz-1]
        laplacian_v[:, -1] = (self.u[:, -2] - self.u[:, -1]) / dz_w_sq[-1] # One-sided difference approx

        visc_force += self.Av * laplacian_v

        return visc_force * self.maskW # Apply only at open west faces

    def _apply_surface_forcing(self, tendency_field, flux_per_area, state_var):
        """ Applies a surface flux to the tendency of the uppermost wet cell. """
        tendency_increment = np.zeros_like(tendency_field)
        recip_wet_area = self.get_recip_wet_area()

        # Determine surface flux value per column
        if callable(flux_per_area):
            surface_flux_values = flux_per_area(self.x_center)
        else:
            surface_flux_values = np.full(self.nx, flux_per_area)

        for i in range(self.nx):
            k_surf = -1
            for k in range(self.nz): # Find topmost wet cell
                if self.maskC[i, k] > 0:
                    k_surf = k
                    break
            if k_surf != -1:
                total_flux_into_cell = surface_flux_values[i] * self.dx[i] # Flux over dx
                # Tendency = Flux / Volume = (Flux*dx) / (wet_area)
                # Units check: (Force*Area_x) / (Area_xz) = Force / Length_z ? Need Force / Mass.
                # Momentum: Flux is Tau (N/m^2). Tendency is m/s^2.
                # Tendency = (Tau * dx) / (rho0 * wet_area[i, k_surf])
                # Tracer: Flux is Q (TracerUnits*m/s). Tendency is TracerUnits/s.
                # Tendency = (Q * dx) / wet_area[i, k_surf]
                if state_var == 'u': # Momentum flux (Stress)
                    # Apply wind stress to u-tendency of the surface layer
                    # Should really apply to u[i, k_surf], but tendency is cell-based
                    # We apply the force over the cell area, distributed to the west face?
                    # Simple: Add forcing to the tendency of the cell containing the surface u-velocity
                     tendency_increment[i, k_surf] = total_flux_into_cell / (self.rho0 * self.wet_area[i, k_surf])
                else: # Tracer flux
                     tendency_increment[i, k_surf] = total_flux_into_cell * recip_wet_area[i, k_surf]

        return tendency_increment

    def _apply_bottom_drag(self):
        """ Applies quadratic bottom drag to u-tendency near the seabed. """
        drag_force = np.zeros_like(self.u_tendency)
        # Find bottom-most u-velocity point for each column i
        # This means finding the lowest k where maskW[i, k] is 1.
        for i in range(1, self.nx): # Drag acts on interior faces
             k_bot = -1
             for k in range(self.nz - 1, -1, -1):
                   # Need west face mask AND cell mask to apply drag
                   if self.maskW[i, k] > 0 and self.maskC[i, k] > 0:
                       k_bot = k
                       break
             if k_bot != -1:
                 # Quadratic drag: F_drag = - C_d * |u| * u / dz_eff
                 # Apply as a tendency = F_drag / rho0 = - C_d * |u| * u / (dz_eff * rho0) ? No rho0 needed?
                 # Tendency = Force / mass = (Stress * Area) / (rho * Volume)
                 # Stress = rho0 * Cd * |u| * u
                 # Force = Stress * dx (acting on face area dx*1)
                 # Tendency = (rho0 * Cd * |u| * u * dx) / (rho0 * wet_area) = Cd * |u| * u * dx / wet_area
                 u_bot = self.u[i, k_bot]
                 drag_stress = self.bottom_drag_coeff * np.abs(u_bot) * u_bot
                 # Apply this drag stress over the relevant cell area
                 # Effective cell height for drag? Use dz[k_bot] * hFacC[i,k_bot]?
                 # Simplification: Apply tendency directly to the face velocity
                 # Tendency contribution = -Cd * |u| * u / H_eff where H is effective layer thickness
                 # Let's use the geometric thickness of the bottom-most cell portion
                 H_eff = self.dz[k_bot] * self.hFacC[i, k_bot]
                 if H_eff > 1e-6:
                    # Apply drag to u_tendency at the face location [i, k_bot]
                    drag_force[i, k_bot] = - self.bottom_drag_coeff * np.abs(u_bot) * u_bot / H_eff

        return drag_force * self.maskW # Ensure only applied at relevant faces


    def calculate_momentum_tendency(self):
        """ Calculates tendency for u-velocity (west face i)."""
        # --- Calculate forcing terms ---
        # Pressure Gradient Force
        pgf = self._calculate_pressure_gradient_force() # Force per unit mass (m/s^2)

        # Viscosity
        visc = self._calculate_viscosity_force()

        # --- Combine terms ---
        # Omitting Advection: u_tendency = pgf + visc + forcing/friction
        self.u_tendency = pgf + visc

        # --- Surface Wind Stress ---
        # Apply wind stress as forcing to the surface layer u-tendency
        # Need to convert Tau (N/m^2) to acceleration (m/s^2)
        # Accel = Tau / (rho0 * H_surface_layer)
        # Find surface layer index k_surf for each i
        # H_surface_layer = dz[k_surf] * hFacC[i, k_surf]?
        # Simplification: Apply forcing based on cell area (done in _apply_surface_forcing)
        wind_tendency = np.zeros_like(self.u_tendency)
        if self.wind_stress_tau_x is not None:
             # We calculate surface forcing per cell, need to apply it to the corresponding WEST face tendency
             surf_force_on_cell = self._apply_surface_forcing(np.zeros_like(self.T), self.wind_stress_tau_x, 'u')
             # Find k_surf for each i and apply surf_force_on_cell[i, k_surf] to u_tendency[i, k_surf]
             for i in range(self.nx):
                  k_surf = -1
                  for k in range(self.nz): # Find topmost wet cell
                      if self.maskC[i, k] > 0:
                          k_surf = k
                          break
                  if k_surf != -1:
                       wind_tendency[i, k_surf] += surf_force_on_cell[i, k_surf]

        self.u_tendency += wind_tendency * self.maskW # Apply only if face is wet

        # --- Bottom Drag ---
        drag_tendency = self._apply_bottom_drag()
        self.u_tendency += drag_tendency # Already masked by maskW inside function

        # Final masking ensures tendency is zero on closed faces
        self.u_tendency *= self.maskW


    def step(self):
        """ Advances the model state by one time step dt. """
        # 1. Update Density and Hydrostatic Pressure
        self.update_density_pressure()

        # 2. Calculate u-momentum tendency (Pressure Gradient, Viscosity, Forcing)
        self.calculate_momentum_tendency()

        # 3. Diagnose vertical velocity W based on current U field (continuity)
        self.diagnose_vertical_velocity()

        # 4. Calculate Tracer Tendencies (Advection + Diffusion)
        self.calculate_tracer_tendency() # Calculates T_tendency, S_tendency

        # 5. Apply external tracer forcing (e.g., surface heat flux) - ADD LATER IF NEEDED
        # T_tendency += self._apply_surface_forcing(self.T_tendency, heat_flux_func, 'T')
        # S_tendency += self._apply_surface_forcing(self.S_tendency, freshwater_flux_func, 'S')

        # 6. Update State Variables (Euler Forward Step)
        self.u += self.u_tendency * self.dt
        self.T += self.T_tendency * self.dt
        self.S += self.S_tendency * self.dt

        # 7. Apply Masks to updated state variables
        self.u *= self.maskW
        self.T *= self.maskC
        self.S *= self.maskC
        # W is diagnosed, already masked. Density/Pressure updated at start.

    # --- Plotting Methods ---
    def plot_grid_bathy_centroids(self, ax=None, show_hfac=False, plot_centroids=True):
        """Plots the grid, bathymetry, optionally hFacC, and centroids."""
        # (Unchanged from original)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        x_fill = np.concatenate(([self.x_face[0]], self.x_face))
        bathy_fill = -self.bathy_func(x_fill)
        ax.plot(self.x_center, self.bathy_depth_c, 'r-', lw=2.0, label='Bathy', zorder=10)
        ax.fill_between(x_fill, bathy_fill, self.z_face[-1]*1.1, color='grey', alpha=0.6, zorder=0)
        x_mesh, z_mesh = np.meshgrid(self.x_face, self.z_face)
        if show_hfac:
            hfac_masked = np.ma.masked_where(self.maskC.T == 0, self.hFacC.T)
            cmap = plt.get_cmap('Blues')
            cmap.set_bad(color='lightgrey', alpha=0.5)
            im = ax.pcolormesh(x_mesh, z_mesh, hfac_masked, cmap=cmap, vmin=0, vmax=1,
                               edgecolors='k', linewidth=0.5, alpha=0.7, shading='flat', zorder=1)
            plt.colorbar(im, ax=ax, label='hFacC', shrink=0.8)
            ax.set_title('Grid, Bathymetry, hFacC, Centroids')
        else:
            for i in range(self.nx):
                for k in range(self.nz):
                    if self.maskC[i,k] > 0:
                        rect = patches.Rectangle((self.x_face[i], self.z_face[k+1]),
                                                 self.dx[i], self.dz[k],
                                                 linewidth=0.5, edgecolor='k', facecolor='none', alpha=0.3, zorder=1)
                        ax.add_patch(rect)
            ax.set_title('Grid, Bathymetry, Centroids')
        if plot_centroids:
            valid_centroids = self.maskC > 0
            ax.plot(self.centroid_x[valid_centroids], self.centroid_z[valid_centroids],
                    'o', color='orange', markersize=3, label='Centroid', zorder=5)
        ax.set_xlabel('Horizontal Distance (x)')
        ax.set_ylabel('Depth (z)')
        ax.set_ylim(self.z_face[-1] * 1.1, self.z_face[0] * 1.1)
        # ax.set_aspect(abs(np.mean(self.dx)) / abs(np.mean(self.dz)) * 0.1)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        return ax

    def plot_field(self, field_name, ax=None, title=None, clabel=None, cmap='viridis', vmin=None, vmax=None):
        """ Plots a model field (T, S, u, w, density, pressure) using pcolormesh
            with flat shading, aligning data with grid cell corners correctly.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        field_data = getattr(self, field_name, None)
        if field_data is None:
            raise ValueError(f"Field '{field_name}' not found in model state.")

        # --- Consistent Corner Coordinates for All Plots ---
        # The corners of ALL cells (tracer, u-face based, w-face based)
        # are defined by the face coordinates.
        x_plot, z_plot = np.meshgrid(self.x_face, self.z_face)
        # Expected shape: (nz + 1, nx + 1) -> (21, 41) in the example

        # --- Determine Data and Mask (shape nx, nz) ---
        mask_data = None
        if field_name in ['T', 'S', 'density', 'pressure']:
            # Data is at cell centers (nx, nz)
            plot_data = field_data # Shape (nx, nz) -> (40, 20)
            mask_data = (self.maskC == 0) # Shape (nx, nz)
        elif field_name == 'u':
            # Data is on west faces (nx, nz) - plotting cell i uses u[i]
            plot_data = field_data # Shape (nx, nz) -> (40, 20)
            mask_data = (self.maskW == 0) # Shape (nx, nz)
        elif field_name == 'w':
            # Data is on bottom faces (nx, nz) - plotting cell k uses w[k]
            plot_data = field_data # Shape (nx, nz) -> (40, 20)
            mask_data = (self.maskS == 0) # Shape (nx, nz)
        else:
            # Default to cell centers if field location unknown
            print(f"Warning: Unknown location for field '{field_name}'. Assuming cell center.")
            plot_data = field_data
            mask_data = (self.maskC == 0)

        if plot_data.shape != (self.nx, self.nz):
             raise ValueError(f"Unexpected shape {plot_data.shape} for field '{field_name}'. Expected {(self.nx, self.nz)}")
        if mask_data.shape != (self.nx, self.nz):
             raise ValueError(f"Unexpected shape {mask_data.shape} for mask of '{field_name}'. Expected {(self.nx, self.nz)}")


        # --- Prepare for Plotting ---
        # Transpose data and mask for plotting: (nz, nx) -> (20, 40)
        plot_data_T = plot_data.T
        mask_data_T = mask_data.T

        # Apply mask
        plot_data_masked = np.ma.masked_where(mask_data_T, plot_data_T)

        # Choose colormap and set 'bad' color for masked areas
        cmap_obj = plt.get_cmap(cmap)
        cmap_obj.set_bad(color='lightgrey', alpha=0.7)

        # --- Plot using pcolormesh with flat shading ---
        # X and Y are corners (nz+1, nx+1)
        # C (plot_data_masked) is data (nz, nx)
        # This matches the requirement for shading='flat'
        im = ax.pcolormesh(x_plot, z_plot, plot_data_masked, cmap=cmap_obj,
                           vmin=vmin, vmax=vmax, shading='flat') # Explicitly use flat
        plt.colorbar(im, ax=ax, label=clabel or field_name, shrink=0.8)

        # --- Overlay bathymetry ---
        ax.plot(self.x_center, self.bathy_depth_c, 'r-', lw=1.5, zorder=10)
        x_fill = np.concatenate(([self.x_face[0]], self.x_face))
        bathy_fill = -self.bathy_func(x_fill)
        ax.fill_between(x_fill, bathy_fill, self.z_face[-1]*1.1, color='grey', alpha=0.5, zorder=9)

        # --- Labels and Limits ---
        ax.set_xlabel('Horizontal Distance (x)')
        ax.set_ylabel('Depth (z)')
        ax.set_ylim(self.z_face[-1] * 1.1, self.z_face[0] * 1.1)
        ax.set_title(title or f"Model Field: {field_name}")
        # Optional: Adjust aspect ratio if needed
        # ax.set_aspect(abs(np.mean(self.dx)) / abs(np.mean(self.dz)) * 0.1)
        return ax

    def save_state(self, filename="model_state.npz"):
        """ Saves key model state variables to a .npz file. """
        np.savez(filename,
                 T=self.T, S=self.S, u=self.u, w=self.w,
                 pressure=self.pressure, density=self.density)
        print(f"Model state saved to {filename}")

    def load_state(self, filename="model_state.npz"):
        """ Loads key model state variables from a .npz file. """
        try:
            data = np.load(filename)
            self.T = data['T'] * self.maskC
            self.S = data['S'] * self.maskC
            self.u = data['u'] * self.maskW
            # W might be diagnosed, so loading might not be desired, but possible
            if 'w' in data: self.w = data['w'] * self.maskS
            if 'pressure' in data: self.pressure = data['pressure'] * self.maskC
            if 'density' in data: self.density = data['density'] * self.maskC
            print(f"Model state loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: State file {filename} not found.")
        except Exception as e:
            print(f"Error loading state from {filename}: {e}")


if __name__ == "__main__":
    # --- Parameters ---
    nx = 40
    nz = 20
    total_width = 40000.0
    total_depth = 3000.0
    x_face = np.linspace(0, total_width, nx + 1)
    z_nom = np.linspace(0, 1, nz + 1)**1.3 # Non-uniform Z
    z_face = -z_nom * total_depth
    z_face = np.sort(z_face)[::-1]

    def bathymetry_profile(x):
        base = 500 + 2000 * (x / total_width)
        ridge = 1500 * np.exp(-((x - total_width*0.6)**2) / (2 * (total_width*0.08)**2))
        depth = base - ridge
        return np.maximum(50.0, depth) # Min depth

    # --- Model Initialization ---
    model = CutCellOceanModel2D_NonUniform_Expanded(
        x_face, z_face, bathymetry_profile,
        dt=100.0,           # Time step
        Kh=5.0, Kv=1e-4,    # Tracer diffusion
        Ah=20.0, Av=1e-3,   # Momentum viscosity
        rho0=1025.0, g=9.81,
        alpha=2e-4, beta=7.4e-4, T0=15.0, S0=35.5,
        wind_stress_tau_x=0.05, # Apply eastward wind stress (N/m^2)
        bottom_drag_coeff=2e-3  # Bottom drag coeff
    )

    # --- Initial Conditions (Example: Stratification) ---
    for k in range(model.nz):
        # Linear stratification
        depth_frac = (model.centroid_z[:, k] - model.z_face[0]) / (model.z_face[-1] - model.z_face[0])
        model.T[:, k] = model.T0 - 10.0 * depth_frac # Colder deeper
        model.S[:, k] = model.S0 + 0.5 * depth_frac # Saltier deeper
    model.T *= model.maskC
    model.S *= model.maskC
    model.update_density_pressure() # Update initial density/pressure

    # --- Simulation Loop ---
    n_steps = 200
    plot_interval = 50
    time = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.ravel()

    print("\nStarting simulation...")
    for n in range(n_steps + 1):
        if n % plot_interval == 0:
            print(f"Step {n}/{n_steps}, Time: {time/3600:.2f} hours")
            # Clear axes for redraw
            for ax in axes: ax.cla()

            # Plot state
            model.plot_field('T', ax=axes[0], title=f'Temperature (degC) at {time/3600:.1f} hr', clabel='degC', cmap='RdYlBu_r')
            model.plot_field('u', ax=axes[1], title=f'U Velocity (m/s)', clabel='m/s', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
            model.plot_field('density', ax=axes[2], title=f'Density (kg/m3)', clabel='kg/m3', cmap='viridis')
            model.plot_field('w', ax=axes[3], title=f'W Velocity (m/s)', clabel='m/s', cmap='RdBu_r', vmin=-5e-4, vmax=5e-4)

            for ax in axes: ax.set_ylim(model.z_face[-1]*1.1, model.z_face[0]*1.1)
            plt.tight_layout()
            plt.pause(0.1) # Allow plot to update

            # Optional: Save state periodically
            # if n > 0: model.save_state(f"model_state_step_{n}.npz")

        if n < n_steps:
            model.step()
            time += model.dt

    print("Simulation complete.")
    plt.show() # Keep final plot open
