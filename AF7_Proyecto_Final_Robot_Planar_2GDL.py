# AF7 - Proyecto Final: Robot planar de 2 GDL


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del eslabón 1
m1 = 23.902
l1 = 0.45
lc1 = 0.091
I1 = 1.266
b1 = 2.288
fc1 = 7.17
fe1 = 8.8

# Parámetros del eslabón 2
m2 = 3.88
l2 = 0.45
lc2 = 0.048
I2 = 0.093
b2 = 0.175
fc2 = 1.734
fe2 = 1.87

g = 9.81

# Término recurrente del modelo
h = m2 * l1 * lc2

print("Parámetros cargados correctamente.")
print(f"h = m2*l1*lc2 = {h:.6f}")



def matriz_inercia(q):
    q1, q2 = q

    M11 = (
        m1*lc1**2
        + m2*l1**2
        + m2*lc2**2
        + 2*h*np.cos(q2)
        + I1 + I2
    )

    M12 = m2*lc2**2 + h*np.cos(q2) + I2
    M22 = m2*lc2**2 + I2

    return np.array([
        [M11, M12],
        [M12, M22]
    ])



def matriz_coriolis(q, qd):
    q1, q2 = q
    qd1, qd2 = qd

    s2 = np.sin(q2)

    return np.array([
        [-2*h*s2*qd2, -h*s2*qd2],
        [ h*s2*qd1,    0.0]
    ])



def vector_gravedad(q):
    q1, q2 = q

    G1 = g * (
        (m1*lc1 + m2*l1)*np.sin(q1)
        + m2*lc2*np.sin(q1 + q2)
    )

    G2 = g * m2*lc2*np.sin(q1 + q2)

    return np.array([G1, G2])



def vector_friccion(qd, tau, k=50.0):
    qd = np.asarray(qd)
    tau = np.asarray(tau)

    # Aproximación suave de sign(qdot)
    signo_suave = np.tanh(k * qd)

    viscosa = np.array([b1, b2]) * qd
    coulomb = np.array([fc1, fc2]) * signo_suave

    # Aproximación del término de fricción estática cerca de qdot = 0
    limite_estatico = np.array([fe1, fe2])
    tau_saturado = np.clip(tau, -limite_estatico, limite_estatico)
    estatica = (1.0 - np.abs(signo_suave)) * tau_saturado

    return viscosa + coulomb + estatica



def torque_motor(t):
    tau1 = 12.0 * np.sin(0.8 * t)
    tau2 = 3.0 * np.sin(1.2 * t)
    return np.array([tau1, tau2])



def modelo_robot(t, x):
    q = x[0:2]
    qd = x[2:4]

    tau = torque_motor(t)
    M = matriz_inercia(q)
    C = matriz_coriolis(q, qd)
    G = vector_gravedad(q)
    F = vector_friccion(qd, tau)

    # M(q) qdd = tau - C(q,qd)qd - G(q) - F(qd)
    qdd = np.linalg.solve(M, tau - C @ qd - G - F)

    return np.concatenate((qd, qdd))



# Condiciones iniciales
x0 = [0.0, 0.0, 0.0, 0.0]

# Tiempo
ti = 0.0
tf = 10.0
t_eval = np.linspace(ti, tf, 2001)

sol = solve_ivp(
    modelo_robot,
    [ti, tf],
    x0,
    t_eval=t_eval,
    method="RK45",
    max_step=0.02,
    rtol=1e-5,
    atol=1e-7
)

if not sol.success:
    raise RuntimeError(sol.message)

t = sol.t
q1 = sol.y[0]
q2 = sol.y[1]
qd1 = sol.y[2]
qd2 = sol.y[3]

q1_deg = np.rad2deg(q1)
q2_deg = np.rad2deg(q2)
qd1_deg = np.rad2deg(qd1)
qd2_deg = np.rad2deg(qd2)

print("Simulación terminada correctamente.")
print(f"Tiempo final: {t[-1]:.1f} s")



plt.figure(figsize=(10, 5))
plt.plot(t, q1_deg, label="q1 - Articulación 1")
plt.plot(t, q2_deg, label="q2 - Articulación 2")
plt.xlabel("Tiempo [s]")
plt.ylabel("Posición angular [°]")
plt.title("Posición de las articulaciones del robot planar de 2 GDL")
plt.grid(True)
plt.legend()
plt.show()



plt.figure(figsize=(10, 5))
plt.plot(t, qd1_deg, label="dq1/dt")
plt.plot(t, qd2_deg, label="dq2/dt")
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad angular [°/s]")
plt.title("Velocidad de las articulaciones del robot planar de 2 GDL")
plt.grid(True)
plt.legend()
plt.show()



print("Rangos obtenidos en el caso de prueba:")
print(f"q1: {q1_deg.min():.2f}° a {q1_deg.max():.2f}°")
print(f"q2: {q2_deg.min():.2f}° a {q2_deg.max():.2f}°")
print(f"dq1/dt: {qd1_deg.min():.2f}°/s a {qd1_deg.max():.2f}°/s")
print(f"dq2/dt: {qd2_deg.min():.2f}°/s a {qd2_deg.max():.2f}°/s")
