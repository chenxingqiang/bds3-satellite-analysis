"""
Physics utilities for BDS-3 MEO satellite analysis
"""

import numpy as np
from typing import Tuple

def calculate_srp_acceleration_prior(epsilon: float, sat_type: str) -> Tuple[float, float, float]:
    """
    Calculate solar radiation pressure acceleration based on the prior model (equations 5-8)
    
    Args:
        epsilon: Sun azimuth angle (degrees)
        sat_type: Satellite type ('CAST' or 'SECM')
        
    Returns:
        Tuple of (acube_D, acube_B, acube_Y) accelerations
    """
    # Convert to radians
    epsilon_rad = np.radians(epsilon)
    
    # Satellite parameters based on Table 3
    if sat_type == 'CAST':
        # CAST satellite parameters
        m = 1750.0  # kg
        # Surface areas (m²)
        A_Z = 5.0
        A_X = 2.5
        A_star_Z = 1.3
        A_star_X = 2.0
        # Reflection coefficients
        alpha_Z = 0.22
        alpha_X = 0.20
        delta_Z = 0.20
        delta_X = 0.21
        rho_Z = 0.58
        rho_X = 0.59
    else:  # SECM
        # SECM satellite parameters
        m = 1850.0  # kg
        # Surface areas (m²)
        A_Z = 5.3
        A_X = 2.4
        A_star_Z = 1.4
        A_star_X = 2.1
        # Reflection coefficients
        alpha_Z = 0.21
        alpha_X = 0.19
        delta_Z = 0.20
        delta_X = 0.20
        rho_Z = 0.59
        rho_X = 0.61
    
    # Solar constant (W/m²)
    Phi_0 = 1367.0
    
    # Speed of light (m/s)
    c = 299792458.0
    
    # Calculate acceleration coefficients (equation 8)
    # a_i^{aδ} = (Phi_0 * A_i / (m * c)) * (alpha_i + delta_i)
    # a_i^{ρ} = (Phi_0 * A_i / (m * c)) * rho_i
    
    a_Z_ad = Phi_0 * A_Z / (m * c) * (alpha_Z + delta_Z)
    a_X_ad = Phi_0 * A_X / (m * c) * (alpha_X + delta_X)
    a_star_Z_ad = Phi_0 * A_star_Z / (m * c) * (alpha_Z + delta_Z)
    a_star_X_ad = Phi_0 * A_star_X / (m * c) * (alpha_X + delta_X)
    
    a_Z_rho = Phi_0 * A_Z / (m * c) * rho_Z
    a_X_rho = Phi_0 * A_X / (m * c) * rho_X
    a_star_Z_rho = Phi_0 * A_star_Z / (m * c) * rho_Z
    a_star_X_rho = Phi_0 * A_star_X / (m * c) * rho_X
    
    # Calculate combined coefficients (equation 7)
    a_C_ad = 0.5 * (a_Z_ad + a_star_X_ad)
    a_S_ad = 0.5 * (a_Z_ad - a_star_X_ad)
    a_A_ad = 0.5 * (a_star_Z_ad - a_star_X_ad)
    
    a_C_rho = 0.5 * (a_Z_rho + a_star_X_rho)
    a_S_rho = 0.5 * (a_Z_rho - a_star_X_rho)
    a_A_rho = 0.5 * (a_star_Z_rho - a_star_X_rho)
    
    # Calculate D direction acceleration (equation 5)
    cos_epsilon = np.abs(np.cos(epsilon_rad))
    sin_epsilon = np.sin(epsilon_rad)
    
    acube_D = (
        -a_C_ad * (cos_epsilon + sin_epsilon + 1.5)
        - a_S_ad * (cos_epsilon - sin_epsilon - 0.75 * np.sin(epsilon_rad)**2 + 1.5)
        - a_A_ad * (1.5 * cos_epsilon * np.cos(epsilon_rad) + np.cos(epsilon_rad))
        - 2 * a_C_rho * (cos_epsilon * np.cos(epsilon_rad)**2 + np.sin(epsilon_rad)**3)
        - 2 * a_S_rho * (cos_epsilon * np.cos(epsilon_rad)**2 - np.sin(epsilon_rad)**3)
        - 2 * a_A_rho * np.cos(epsilon_rad)**3
    )
    
    # Calculate B direction acceleration (equation 6)
    acube_B = (
        -0.75 * a_S_ad * (np.cos(epsilon_rad) * np.sin(epsilon_rad))
        -1.5 * a_A_ad * (cos_epsilon * np.sin(epsilon_rad))
        -2 * a_C_rho * (cos_epsilon - sin_epsilon) * np.cos(epsilon_rad) * np.sin(epsilon_rad)
        -2 * a_S_rho * (cos_epsilon + sin_epsilon) * np.cos(epsilon_rad) * np.sin(epsilon_rad)
        -2 * a_A_rho * np.cos(epsilon_rad)**2 * np.sin(epsilon_rad)
    )
    
    # Y direction acceleration is approximately zero due to satellite symmetry
    acube_Y = 0.0
    
    # Convert to nm/s²
    return acube_D * 1e9, acube_B * 1e9, acube_Y * 1e9

def apply_srp_model(orbit_angle: float, sun_elevation: float, 
                   d_params: np.ndarray, y_params: np.ndarray, 
                   b_params: np.ndarray, sat_type: str) -> Tuple[float, float, float]:
    """
    Apply the combined SRP model (simplified ECOMC + prior cube model)
    
    Args:
        orbit_angle: Satellite orbit angle (degrees)
        sun_elevation: Sun elevation angle (degrees)
        d_params: D direction parameters [D0, Dc, D2c, D4c]
        y_params: Y direction parameters [Y0, Ys]
        b_params: B direction parameters [B0, Bs]
        sat_type: Satellite type ('CAST' or 'SECM')
        
    Returns:
        Tuple of (a_D, a_Y, a_B) combined accelerations
    """
    # Compute ECOMC accelerations (equation 4)
    orbit_angle_rad = np.radians(orbit_angle)
    
    # Extract parameters
    D0, Dc, D2c, D4c = d_params
    Y0, Ys = y_params
    B0, Bs = b_params
    
    # ECOMC accelerations
    a_D_ecomc = D0 + Dc * np.cos(orbit_angle_rad) + D2c * np.cos(2 * orbit_angle_rad) + D4c * np.cos(4 * orbit_angle_rad)
    a_Y_ecomc = Y0 + Ys * np.sin(orbit_angle_rad)
    a_B_ecomc = B0 + Bs * np.sin(orbit_angle_rad)
    
    # Compute prior model accelerations
    # Simplified calculation based on sun azimuth (approximated from orbit angle)
    sun_azimuth = orbit_angle  # Simplified approximation
    a_D_prior, a_B_prior, a_Y_prior = calculate_srp_acceleration_prior(sun_azimuth, sat_type)
    
    # Eclipse period detection (|β| < 12.9°)
    is_eclipse = abs(sun_elevation) < 12.9
    
    # Combined model - during eclipse, use a blend of ECOMC and prior model
    if is_eclipse:
        # Calculate eclipse factor (0 to 1) based on beta angle
        # 0 means full eclipse, 1 means no eclipse
        eclipse_factor = abs(sun_elevation) / 12.9
        
        # Blend models based on eclipse factor
        blend_factor = 0.7  # Weight for ECOMC model
        
        a_D = blend_factor * a_D_ecomc + (1 - blend_factor) * a_D_prior
        a_Y = blend_factor * a_Y_ecomc + (1 - blend_factor) * a_Y_prior
        a_B = blend_factor * a_B_ecomc + (1 - blend_factor) * a_B_prior
    else:
        # Outside eclipse, primarily use ECOMC model
        a_D = a_D_ecomc
        a_Y = a_Y_ecomc
        a_B = a_B_ecomc
    
    return a_D, a_Y, a_B
