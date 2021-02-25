import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf
import tensorflow_probability as tfp
import subprocess as sp
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import math

def readWaveTime(caseDir):
	caseDirName = caseDir+"/outputFlowHeight.csv"
	df = pd.read_csv(caseDirName)
	waveTime = df[df['outputFlowHeight']>=1e-04].iloc[0,0]
	return waveTime

def calcLoss(coeffs, referenceTimeValue=1.2, caseDir="constantAngleSlopeTurbKE_U5e-1"):
	sp.call("cd "+caseDir+";\
			bash -i of1912;\
			bash -i ./Allclean;\
			sed \"s/Cmu_pattern/	Cmu			"+str(coeffs[0])+";/\" constant/turbulenceProperties > tmp;\
			mv tmp constant/turbulenceProperties;\
			bash -i ./Allrun;\
			cd ../", shell=True)
#			sed \"s/C1_pattern/	C1			"+str(coeffs[1])+";/\" constant/turbulenceProperties > tmp;\
#			mv tmp constant/turbulenceProperties;\
#			sed \"s/C2_pattern/	C2			"+str(coeffs[2])+";/\" constant/turbulenceProperties > tmp;\
#			mv tmp constant/turbulenceProperties;\
#			sed \"s/C3_pattern/	C3			"+str(coeffs[3])+";/\" constant/turbulenceProperties > tmp;\
#			mv tmp constant/turbulenceProperties;\
#			sed \"s/sigmak_pattern/	sigmak		"+str(coeffs[4])+";/\" constant/turbulenceProperties > tmp;\
#			mv tmp constant/turbulenceProperties;\
#			sed \"s/sigmaEps_pattern/	sigmaEps	"+str(coeffs[5])+";/\" constant/turbulenceProperties > tmp;\
#			mv tmp constant/turbulenceProperties;\
	TurbKETime = readWaveTime(caseDir)
	print("TurbKETime = " + str(TurbKETime))
#	TurbKETime = tf.reduce_sum(coeffs**2, axis=-1)
#	TurbKETime = sum(coeffs**2)
#	print(referenceTimeValue - TurbKETime)
#	return referenceTimeValue - TurbKETime
	print(coeffs)
	return TurbKETime

def runSGD(initCoeffs, referenceValue=1.2, caseDir="constantAngleSlopeTurbKE_U5e-1"):
	opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
	var = tf.Variable(initCoeffs[0])
	val0 = var.value()
	loss = calcLoss(caseDir, var, referenceValue) #lambda: (var ** 2)/2.0         # d(loss)/d(var1) = var1
	# First step is `- learning_rate * grad`
	step_count = opt.minimize(loss, [var]).numpy()
	val1 = var.value()
	print(val0 - val1).numpy()

#	# On later steps, step-size increases because of momentum
#	step_count = opt.minimize(loss, [var]).numpy()
#	val2 = var.value()
#	(val1 - val2).numpy()

def runNelderMead(initCoeffs, referenceValue=1.2, caseDir="constantAngleSlopeTurbKE_U5e-1"):
	start = tf.constant(initCoeffs)  # Starting point for the search.
	optim_results = tfp.optimizer.nelder_mead_minimize(
		calcLoss, initial_vertex=start, func_tolerance=1e-8,
		batch_evaluate_objective=True)

	# Check that the search converged
	assert(optim_results.converged)
	# Check that the argmin is close to the actual value.
	np.testing.assert_allclose(optim_results.position, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-7)
	# Print out the total number of function evaluations it took.
	print("Function evaluations: %d" % optim_results.num_objective_evaluations)

def testNelderMead():
	# The objective function
	def sqrt_quadratic(x):
		return tf.reduce_sum(x ** 2, axis=-1)

	start = tf.Variable([0.09, 1.44, 1.92, 0.0, 1.00, 1.3])  # Starting point for the search.
	optim_results = tfp.optimizer.nelder_mead_minimize(
		sqrt_quadratic, initial_vertex=start, func_tolerance=1e-4,
		batch_evaluate_objective=True)

	# Check that the search converged
	assert(optim_results.converged)
	# Check that the argmin is close to the actual value.
	np.testing.assert_allclose(optim_results.position, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-3)
	# Print out the total number of function evaluations it took.
	print("Function evaluations: %d" % optim_results.num_objective_evaluations)

def minimizeTFP(initCoeffs, referenceValue, caseDir):
#	coeffs = initCoeffs
#	coeffs = tf.Variable([0.09, 1.44, 1.92, 0.00, 1.00, 1.3])
	with tf.control_dependencies([losses]):
		optimized_coeffs = tf.identity(initCoeffs)  # Use a dummy op to attach the dependency.
	losses = tfp.math.minimize(calcLoss, num_steps=100, optimizer=tf.optimizers.Adam(learning_rate=0.1))
	# In TF2/eager mode, the optimization runs immediately.
	print("optimized value is {} with loss {}".format(initCoeffs, losses[-1]))

def minimizeSciPy(initCoeffs, referenceValue, caseDir):
	res = minimize(calcLoss, initCoeffs, args=(referenceValue, caseDir), method='Nelder-Mead', tol=1e-6)
	print(res.x)

def main():
#	coeffs = [0.09, 1.44, 1.92, 0.00, 1.00, 1.3]
	coeffs = [0.09]
	DNSCaseDir = "../constantAngleSlopeDNS_U5e-1"
	TurbKECaseDir = "constantAngleSlopeTurbKE_U5e-1"
	DNSTime = readWaveTime(DNSCaseDir)
	print("DNSTime = " + str(DNSTime))
#	print(calcLoss(coeffs, DNSTime, TurbKECaseDir))
#	runSGD(coeffs, DNSTime, TurbKECaseDir)
#	runNelderMead(coeffs, DNSTime, TurbKECaseDir)
#	testNelderMead()
	minimizeSciPy(coeffs, DNSTime, TurbKECaseDir)

if __name__ == "__main__":
	main()
