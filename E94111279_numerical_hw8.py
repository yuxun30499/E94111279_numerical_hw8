import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad
import matplotlib.pyplot as plt

print("==== 第一題：資料擬合 ====")
# 題目1資料
x1 = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3])
y1 = np.array([102.6, 113.2, 130.1, 142.1, 167.5, 195.1, 224.9, 256.8])

# 1.a 二次多項式擬合
coeffs_poly = np.polyfit(x1, y1, 2)
y1_pred_poly = np.polyval(coeffs_poly, x1)
error_poly = np.sum((y1 - y1_pred_poly) ** 2)
print("1.a 二次多項式係數 (c2, c1, c0):", coeffs_poly)
print("1.a 擬合誤差:", error_poly)

# 1.b 指數函數擬合 y = b * exp(a * x)
def exp_func(x, a, b):
    return b * np.exp(a * x)

popt_exp, _ = curve_fit(exp_func, x1, y1, p0=(0.1, 1))
y1_pred_exp = exp_func(x1, *popt_exp)
error_exp = np.sum((y1 - y1_pred_exp) ** 2)
print("\n1.b 指數函數擬合參數 (a, b):", popt_exp)
print("1.b 擬合誤差:", error_exp)

# 1.c 冪次函數擬合 y = b * x^n
def power_func(x, n, b):
    return b * np.power(x, n)

popt_pow, _ = curve_fit(power_func, x1, y1, p0=(1, 1))
y1_pred_pow = power_func(x1, *popt_pow)
error_pow = np.sum((y1 - y1_pred_pow) ** 2)
print("\n1.c 冪次函數擬合參數 (n, b):", popt_pow)
print("1.c 擬合誤差:", error_pow)

print("\n==== 第二題：區間 [-1,1] 上的二次多項式擬合 ====")
# 定義第二題函數和基底
def f2(x):
    return 0.5 * np.cos(x) + 0.25 * np.sin(2 * x)

def phi0(x): return 1
def phi1(x): return x
def phi2(x): return x**2

phi = [phi0, phi1, phi2]

from scipy.integrate import quad

def A_ij(i, j):
    integrand = lambda x: phi[i](x) * phi[j](x)
    val, _ = quad(integrand, -1, 1)
    return val

A2 = np.array([[A_ij(i, j) for j in range(3)] for i in range(3)])

def b_i(i):
    integrand = lambda x: f2(x) * phi[i](x)
    val, _ = quad(integrand, -1, 1)
    return val

b2 = np.array([b_i(i) for i in range(3)])

a2 = np.linalg.solve(A2, b2)

print("二次多項式擬合係數 a0, a1, a2:", a2)

print("\n==== 第三題：離散最小平方三角多項式 S4 ====")
m = 16
x3_points = np.linspace(0, 1, m, endpoint=False)
f3 = lambda x: x**2 * np.sin(x)

f3_vals = f3(x3_points)
N = 4

a3 = np.zeros(N + 1)
b3 = np.zeros(N + 1)

a3[0] = (2 / m) * np.sum(f3_vals)

for k in range(1, N + 1):
    a3[k] = (2 / m) * np.sum(f3_vals * np.cos(k * np.pi * x3_points))
    b3[k] = (2 / m) * np.sum(f3_vals * np.sin(k * np.pi * x3_points))

print("a係數:", a3)
print("b係數:", b3)

def S4(x):
    res = a3[0] / 2
    for k in range(1, N + 1):
        res += a3[k] * np.cos(k * np.pi * x) + b3[k] * np.sin(k * np.pi * x)
    return res

integral_S4, _ = quad(S4, 0, 1)
print("∫0^1 S4(x) dx =", integral_S4)

integral_f3, _ = quad(f3, 0, 1)
print("∫0^1 x^2 sin(x) dx =", integral_f3)

error_S4 = np.sum((f3_vals - S4(x3_points)) ** 2)
print("誤差 E(S4) =", error_S4)

