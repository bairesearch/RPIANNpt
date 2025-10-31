import matplotlib.pyplot as plt
import numpy as np

def subset(arr, subsetMask):
	arr = np.array(arr)[np.array(subsetMask)]
	return arr
	
graphedDatasets = [False, False, True, True, True, True, True, True, True, True, True, True, False, False, True] 	#tested datasets
#
y1_values = [95.82, 91.59, 62.86, 70.17, 90.15, 58.13, 91.41, 61.23, 85.54, 85.77, 92.14, 66.85, 0.0, 0.0, 95.45]		#ANN full backprop (from AEANNpt codebase)
y2_values = [0.0, 0.0, 61.50, 70.19, 88.54, 55.62, 100.00, 60.68, 88.50, 84.27, 85.78, 65.72, 0.0, 0.0, 90.90]		#RPIANN useClassificationLayerLoss=True + inputProjectionActivationFunction=False
#y1_values = [95.82, 91.67, 64.95, 70.11, 89.39, 58.40, 93.36, 62.56, 83.97, 85.63, 92.47, 65.91, 71.73, 100.0, 95.45]		#ANN full backprop (from AEANNpt codebase)
#y2_values = [0.0, 0.0, 61.56, 70.09, 87.46, 54.10, 91.80, 60.99, 87.05, 83.47, 76.09, 64.48, 48.88, 100.0, 82.81]		#RPIANN

#y1_values = [0.0, 0.0, 61.88, 70.22, 90.69, 51.25, 96.48, 58.59, 87.34, 84.35, 88.57, , , ]	#RPIANN useClassificationLayerLoss=True + hiddenLayerSizeHigh=False + numberOfSublayers=2 + subLayerFirstNotTrained=False  with full backprop
#y2_values = [0.0, 0.0, 62.26, 70.19, 89.68, 44.37, 92.58, 59.12, 88.66, 84.93, 87.12, 62.11, 90.99]		#RPIANN useClassificationLayerLoss=True + hiddenLayerSizeHigh=False + numberOfSublayers=2 + subLayerFirstNotTrained=False 
#y2_values = [0.0, 0.0, 60.83, 70.18, 88.31, 43.75, 98.05, 58.24, 88.35, 85.26, , , ]		#RPIANN useClassificationLayerLoss=True + inputProjectionActivationFunction=True 
#y2_values = [0.0, 0.0, 61.38, 69.88, 86.50, 61.25, 72.66, 58.17, 85.92, 83.68, 83.74, 64.22, 90.90]		#RPIANN useClassificationLayerLoss=True + numberOfLayers=1 (ie lay=1) 
#y2_values = [0.0, 0.0, 59.51, 69.75, 87.86, 56.87, 76.76, 59.83, 87.23, 85.24, 89.44, , ]		#RPIANN useClassificationLayerLoss=True + numberOfLayers=1 (ie lay=1) + trainNumberOfEpochsHigh 
#y2_values = [0.0, 0.0, 60.51, 69.18, 82.41, 60.62, 95.12, 53.03, 89.35, 80.39, , , ]		#RPIANN useRecursiveLayers=False 
#y2_values = [0.0, 0.0, 61,95, 70.00, 83.58, 49.37, 92.97, 51.88, 88.37, 80.32, , ,]		#RPIANN useRecursiveLayers=False + numberOfLayers=1 (ie lay=1) 
#y2_values = [0.0, 0.0, 62.19, 69.75, 88.23, 59.38, 94.92, 59.65, 88.19, 83.74, , , ]		#RPIANN useRecursiveLayers=False + numberOfLayers=1 (ie lay=1) + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 61.86, 70.17, 88.77,51.25, 89.06, 59.24, 88.73, 84.29, , ,]		#RPIANN numberOfSublayers=2 + numberOfLayers=1 (ie lay=1) + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 60.08, 69.60, 88.24, 45.00, 78.52, 54.67, 88.42, 78.64, , ,]		#RPIANN numberOfSublayers=2 + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 52.15, 70.02, 86.54, 66.25, 96.88, 45.88, 89.15, 84.38, , , ]		#RPIANN layersFeedConcatInput=False + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 60.32, 70.12, 86.56, 56.87, 96.09, 58.09, 89.13, 84.75, , , ]		#RPIANN layersFeedConcatInput=False + useRecursiveLayers=False + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 61.09, 69.84, 86.20, 40.62, 83.98, 59.60, 88.88, 83.35, , , ]		#RPIANN layersFeedConcatInput=False + numberOfLayers=1 (ie lay=1) + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 60.68, 70.12, 86.09, 56.25, 81.64, 59.56, 88.86, 85.05, , , ]		#RPIANN layersFeedConcatInput=False+layersFeedResidualInput=True + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 62.12, 70.05, 87.23, 62.50, 91.60, 59.66, 89.04, 83.78, , , ]		#RPIANN layersFeedConcatInput=False+layersFeedResidualInput=True + useRecursiveLayers=False + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 60.56, 69.88, 85.86, 53.75, 76.37, 59.79, 88.37, 83.78, , , ]		#RPIANN layersFeedConcatInput=False+layersFeedResidualInput=True + numberOfLayers=1 (ie lay=1) + targetProjectionActivationFunction=False
#y2_values = [0.0, 0.0, 59.70, 70.18, 87.97, 64.38, 100.00, 54.12, 89.20, 84.40, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0
#y2_values = [0.0, 0.0, 58.10, 70.19, 87.39, 51.88, 69.73, 54.12, 89.00, 84.38, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + useRecursiveLayers=False
#y2_values = [0.0, 0.0, 59.03, 70.22, 87.45, 55.00, 96.29, 54.13, 88.73, 83.22, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + numberOfLayers=1 (ie lay=1)
#y2_values = [0.0, 0.0, 57.42, 70.18, 87.15, 51.88, 89.26, 54.12, 88.55, 83.10, , ,]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + targetProjectionActivationFunction=False+targetProjectionActivationFunctionTanh=True
#y2_values = [0.0, 0.0, , , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + targetProjectionActivationFunction=False+targetProjectionActivationFunctionTanh=True + useRecursiveLayers=False
#y2_values = [0.0, 0.0, , , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + targetProjectionActivationFunction=False+targetProjectionActivationFunctionTanh=True + numberOfLayers=1 (ie lay=1) 
#y2_values = [0.0, 0.0, 55.89, 70.23, 88.30, 48.75, 82.62, 54.12, 88.68, 84.70, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + targetProjectionActivationFunctionTanh=True
#y2_values = [0.0, 0.0, 59.62, 69.73, 88.05, 61.87, 96.68, 54.12, 88.91, 84.19, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + targetProjectionActivationFunctionTanh=True + useRecursiveLayers=False
#y2_values = [0.0, 0.0, 59.10, 70.19, 87.63, 57.50, 86.91, 54.12, 88.73, 84.00, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + targetProjectionActivationFunctionTanh=True + numberOfLayers=1 (ie lay=1) 
#y2_values = [0.0, 0.0, 51.01, 70.18, 87.09, 53.75, 78.52, 54.13, 89.11, 84.40, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + inputProjectionActivationFunctionTanh=True+targetProjectionActivationFunctionTanh=True
#y2_values = [0.0, 0.0, 58.29, 70.15, 88.47, 54.37, 95.12, 54.12, 89.15, 75.04, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + inputProjectionActivationFunctionTanh=True+targetProjectionActivationFunctionTanh=True + useRecursiveLayers=False
#y2_values = [0.0, 0.0, 52.54, 70.19, 87.71, 54.37, 94.73, 54.13, 89.26, 80.82, , , ]		#RPIANN layersFeedConcatInput=False + layerScale=1.0 + inputProjectionActivationFunctionTanh=True+targetProjectionActivationFunctionTanh=True + numberOfLayers=1 (ie lay=1) 


group_labels = ["CIFAR-10 Resnet-18", "CIFAR-10 Conv-9", "tabular-benchmark", "blog-feedback", "titanic", "red-wine", "breast-cancer-wisconsin", "diabetes-readmission", "banking-marketing", "adult_income_dataset", "covertype", "higgs", "topquark", "iris", "new-thyroid"]

group_descriptions = [
	"CIFAR-10: Resnet-18; lay=16, units<=65536 (ch*pix)\n                lin lay=1, units=512",
	"CIFAR-10: Conv-9; lay=6, units=12288 (ch*pix)\n                lin lay=2, units=1024",
	"tabular-benchmark: MLP-5; lay=3, units=64",
	"blog-feedback: MLP-5; lay=3, units=144",
	"titanic: MLP-5; lay=3, units=128",
	"red-wine: MLP-5; lay=3, units=128",
	"breast-cancer-wisconsin: MLP-5; lay=3, units=32",
	"diabetes-readmission: MLP-5; lay=3, units=304",
	"banking-marketing: MLP-6; lay=4, units=128",
	"adult_income_dataset: MLP-5; lay=3, units=256",
	"covertype: MLP-7; lay=5, units=512",
	"higgs: MLP-5; lay=3, units=256",
	"topquark: MLP-6; lay=4, units=256",
	"iris: MLP-4; lay=2, units=32",
	"new-thyroid: MLP-4; lay=2, units=16",
]

# Convert to arrays
x = np.arange(len(subset(y1_values, graphedDatasets)))
y1 = np.array(subset(y1_values, graphedDatasets))
y2 = np.array(subset(y2_values, graphedDatasets))
width = 0.4

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars
b1 = ax.bar(x, y1, width=width, color='blue', label='Full Backprop training')
b2 = ax.bar(x + width, y2, width=width, color='magenta', label='Recursive Prediction Improvement training')

# Annotate bars
ax.bar_label(b1, fmt='%.1f', padding=3, fontsize=6)
ax.bar_label(b2, fmt='%.1f', padding=3, fontsize=6)

# X-axis group labels
ax.set_xticks(x + width/2)
ax.set_xticklabels(subset(group_labels, graphedDatasets), rotation=45, ha='right', fontsize=10)

# Minor ticks
#ax.minorticks_on()
#ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax.set_yticks(np.arange(0, 100+0.1, 10.0))

# Labels and title
ax.set_xlabel("dataset")
ax.set_ylabel("test accuracy (Top-1)")
ax.set_title("Recursive Prediction Improvement vs Full Backprop Training")

# Legend of series, moved to the right above the set descriptions
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.95), borderaxespad=0.)

# Adjust layout to make room for both legends and description key on the right
fig.subplots_adjust(right=0.8)

# Add descriptive key box on the right
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
key_text = "\n".join(subset(group_descriptions, graphedDatasets))
ax.text(1.02, 0.5, key_text, transform=ax.transAxes, fontsize=10, verticalalignment='center', bbox=props)

plt.tight_layout()
plt.show()
