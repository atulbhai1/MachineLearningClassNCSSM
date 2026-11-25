import numpy as np
import math
#Just a collection of lambda functions for use in customKNNframework.py

linear = lambda p, n, x: 1/x
quadratic = lambda p, n, x: 1/(x**2)
cubic = lambda p, n, x: 1/(x**3)
quartic = lambda p, n, x: 1/(x**4)
quintic = lambda p, n, x: 1/(x**5)
sextic = lambda p, n, x: 1/(x**6)
septic = lambda p, n, x: 1/(x**7)
octic = lambda p, n, x: 1/(x**8)
nonic = lambda p, n, x: 1/(x**9)
decic = lambda p, n, x: 1/(x**10)

uniform = lambda p, n, x: 1#Baseline: every neighbor gets the same weight (plain majority vote)

log = lambda p, n, x: 1/(np.log1p(x))

sqrt = lambda p, n, x: 1/(np.sqrt(x))
cubic_root = lambda p, n, x: 1/(np.cbrt(x))
quartic_root = lambda p, n, x: 1/(np.power(x, 1/4))
quintic_root = lambda p, n, x: 1/(np.power(x, 1/5))
sextic_root = lambda p, n, x: 1/(np.power(x, 1/6))
septic_root = lambda p, n, x: 1/(np.power(x, 1/7))
octic_root = lambda p, n, x: 1/(np.power(x, 1/8))
nonic_root = lambda p, n, x: 1/(np.power(x, 1/9))
decic_root = lambda p, n, x: 1/(np.power(x, 1/10))
doesntHandle0List = [linear, quadratic, cubic, quartic, quintic, sextic, septic, octic, nonic, decic, uniform, log, sqrt, cubic_root, quartic_root, quintic_root, sextic_root, septic_root, octic_root, nonic_root, decic_root]
for item in doesntHandle0List:
    item.handles0distance = False


handlesLinear = lambda p, n, x: 1/(x+1)
handlesQuadratic = lambda p, n, x: 1/((x+1)**2)
handlesCubic = lambda p, n, x: 1/((x+1)**3)
handlesQuartic = lambda p, n, x: 1/((x+1)**4)
handlesQuintic = lambda p, n, x: 1/((x+1)**5)
handlesSextic = lambda p, n, x: 1/((x+1)**6)
handlesSeptic = lambda p, n, x: 1/((x+1)**7)
handlesOctic = lambda p, n, x: 1/((x+1)**8)
handlesNonic = lambda p, n, x: 1/((x+1)**9)
handlesDecic = lambda p, n, x: 1/((x+1)**10)

handlesLog = lambda p, n, x: 1/(1+np.log1p(x))

handlesSqrt = lambda p, n, x: 1/np.power(x+1, 1/2)
handlesCubicRoot = lambda p, n, x: 1/np.power(x+1, 1/3)
handlesQuarticRoot = lambda p, n, x: 1/np.power(x+1, 1/4)
handlesQuinticRoot = lambda p, n, x: 1/np.power(x+1, 1/5)
handlesSexticRoot = lambda p, n, x: 1/np.power(x+1, 1/6)
handlesSepticRoot = lambda p, n, x: 1/np.power(x+1, 1/7)
handlesOcticRoot = lambda p, n, x: 1/np.power(x+1, 1/8)
handlesNonicRoot = lambda p, n, x: 1/np.power(x+1, 1/9)
handlesDecicRoot = lambda p, n, x: 1/np.power(x+1, 1/10)

handlesList = [handlesLinear, handlesQuadratic, handlesCubic, handlesQuartic, handlesQuintic, handlesSextic, handlesSeptic, handlesOctic, handlesNonic, handlesDecic, handlesLog, handlesSqrt, handlesCubicRoot, handlesQuarticRoot, handlesQuinticRoot, handlesSexticRoot, handlesSepticRoot, handlesOcticRoot, handlesNonicRoot, handlesDecicRoot]
for item in handlesList:
    item.handles0distance = True