import numpy as np
from scipy.optimize import fsolve

def exact_sod(t, gamma=1.4):
    # Left state
    rho_l = 1.0
    u_l = 0.0
    p_l = 1.0
    c_l = np.sqrt(gamma * p_l / rho_l)

    # Right state
    rho_r = 0.125
    u_r = 0.0
    p_r = 0.1
    c_r = np.sqrt(gamma * p_r / rho_r)

    # Solve for p3 (pressure in the star region)
    def shock_tube_eq(p3):
        # Rarefaction wave on the left
        z = (p3 / p_l) ** ((gamma - 1) / (2 * gamma))
        u3 = u_l + 2 * c_l / (gamma - 1) * (1 - z)
        # Shock wave on the right
        A = 2 / ((gamma + 1) * rho_r)
        B = (gamma - 1) / (gamma + 1) * p_r
        u3_r = u_r + (p3 - p_r) * np.sqrt(A / (p3 + B))
        return u3 - u3_r

    p3 = fsolve(shock_tube_eq, 0.3)[0]
    
    # Calculate all states and wave speeds
    # Region 3 (left of contact, behind rarefaction)
    u3 = u_l + 2 * c_l / (gamma - 1) * (1 - (p3 / p_l) ** ((gamma - 1) / (2 * gamma)))
    rho3 = rho_l * (p3 / p_l) ** (1 / gamma)
    c3 = np.sqrt(gamma * p3 / rho3)
    
    # Region 4 (right of contact, behind shock)
    rho4 = rho_r * (p3 / p_r + (gamma - 1) / (gamma + 1)) / (1 + (gamma - 1) / (gamma + 1) * p3 / p_r)
    u4 = u3 # Contact discontinuity
    
    # Speeds
    # Head and tail of rarefaction
    v_head = u_l - c_l
    v_tail = u3 - c3
    
    # Contact discontinuity
    v_contact = u3
    
    # Shock wave
    v_shock = u_r + c_r * np.sqrt((gamma + 1) / (2 * gamma) * (p3 / p_r) + (gamma - 1) / (2 * gamma))
    
    print(f"v_head = {v_head}")
    print(f"v_tail = {v_tail}")
    print(f"v_contact = {v_contact}")
    print(f"v_shock = {v_shock}")
    print(f"rho1 (L) = {rho_l}")
    print(f"rho3 (star L) = {rho3}")
    print(f"rho4 (star R) = {rho4}")
    print(f"rho5 (R) = {rho_r}")
    
    # Rarefaction region as function of x, t
    # u_fan = 2/(g+1) * (c_l + (x-x0)/t)
    # rho_fan = rho_l * (1 - (g-1)/2 * u_fan / c_l)**(2/(g-1))
    # Let x0 = 0
    # For sympy:
    print("\nSympy Piecewise for rho at t=0.25:")
    t_val = t
    pos_head = v_head * t_val
    pos_tail = v_tail * t_val
    pos_contact = v_contact * t_val
    pos_shock = v_shock * t_val
    
    # Rarefaction expression for rho
    # rho_fan = rho_l * ( 2/(g+1) + (g-1)/((g+1)*c_l) * (u_l - x/t) )**(2/(g-1))
    # Wait, the classical rarefaction wave:
    # rho(x,t) = rho_l * [ 2/(g+1) + (g-1)/(g+1) * (u_l - x/t)/c_l ] ** (2/(g-1))
    
    # For rho(x, 0.25)
    print("Piecewise(")
    print(f"    ({rho_l}, x <= {pos_head}),")
    print(f"    ( {rho_l} * (2/2.4 - 0.4/(2.4*{c_l}) * (x/{t_val}))**(2/0.4), x <= {pos_tail} ),")
    print(f"    ({rho3}, x <= {pos_contact}),")
    print(f"    ({rho4}, x <= {pos_shock}),")
    print(f"    ({rho_r}, True)")
    print(")")

exact_sod(0.25)
