import re
import pandas as pd

patternTime = re.compile("Time =")
patternOutputFlowHeight = re.compile("alpha.water =")
patternFloatDigit = re.compile(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')

df = pd.DataFrame()

time = 0.0
outputFlowHeight = 0.0

for line in open("log.postProcess"):
	if (re.search(patternTime, line)):
		time = float(re.search(patternFloatDigit, line).group())
	if (re.search(patternOutputFlowHeight, line)):
		outputFlowHeight = float(re.search(patternFloatDigit, line).group()) / 5e-03 
		df = df.append([[time, outputFlowHeight]], ignore_index=True)
df = df.rename(columns={0: "time", 1: "outputFlowHeight"})
df.to_csv("outputFlowHeight.csv", index=False)
