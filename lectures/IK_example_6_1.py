import numpy as np
import modern_robotics as mr

def main():
    M = np.array([[1, 0, 0, 2],  
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    Blist = np.array([[0, 0, 1, 0, 2, 0],  
                      [0, 0, 1, 0, 1, 0]]).T  

    
    theta0_deg = (0, 30)
    theta0_rad = np.radians(theta0_deg)

    Tsd = np.array([[-0.985,  0.174, 0, -0.811],
                    [-0.174, -0.985, 0,  0.811],
                    [ 0,      0,     1,  0    ],
                    [ 0,      0,     0,  1    ]])

    epsilon_omega = 0.001  
    epsilon_v = 1e-4      

  
    print("\nNewton-Raphson Iterations for 2R Robot:")
    print("===========================================\n")
    theta_sol, success = mr.IKinBody(Blist, M, Tsd, theta0_rad, 
                                    epsilon_omega, epsilon_v)

    print("\nFinal Result:")
    print(f"Success: {success}")
    print(f"Joint Angles (rad): {theta_sol}")
    print(f"Joint Angles (°): {np.degrees(theta_sol)}")

if __name__ == "__main__":
    main()