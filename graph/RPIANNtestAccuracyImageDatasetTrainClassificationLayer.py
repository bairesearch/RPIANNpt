import matplotlib.pyplot as plt
import numpy as np

def subset(arr, subsetMask):
	arr = np.array(arr)[np.array(subsetMask)]
	return arr

showXaxisUniform = True

# groups to plot
graphedDatasets = [True, True, True, True, True]

x_values = [512, 2048, 8192]
if(showXaxisUniform):
	x_pos = np.arange(len(x_values))

# One label per group (these are also the legend keys lines)
group_labels = ["numberOfLayers=9 [N/A]", "numberOfLayers=9 trainLocal=False", "numberOfLayers=1r9", "numberOfLayers=1r9 trainLocal=False", "numberOfLayers=1"]
	
# Optional long descriptions shown in the right-side key
group_descriptions = [
	"",		#N/A
	"CNNprojectionNumlayers=1 numberOfLayers=9 inp+targetProjectionActivationFunctionTanh layersFeedConcatInput initialiseYhatZero trainLocal=False useClassificationLayerLoss trainClassificationLayer",
	"CNNprojectionNumlayers=1 numberOfLayers=9 inp+targetProjectionActivationFunctionTanh layersFeedConcatInput initialiseYhatZero useRecursiveLayers useClassificationLayerLoss trainClassificationLayer",
	"CNNprojectionNumlayers=1 numberOfLayers=9 inp+targetProjectionActivationFunctionTanh layersFeedConcatInput initialiseYhatZero useRecursiveLayers trainLocal=False useClassificationLayerLoss trainClassificationLayer",
	"CNNprojectionNumlayers=1 numberOfLayers=1 inp+targetProjectionActivationFunctionTanh layersFeedConcatInput initialiseYhatZero (useRecursiveLayers) useClassificationLayerLoss trainClassificationLayer",
]

y_values_lists = [	
	[0.0, 0.0, 0.0],	#N/A
	[0.4220, 0.5674, 0.6271],	#replications: [0.4442, 0.5852, 0.6303], @8192h: 0.6316
	[0.4802, 0.4903, 0.5526],	#replications: [0.4539, 0.5055, 0.5455], @8192h: 0.5312
	[0.4056, 0.5765, 0.6346],	#replications: [0.4373, 0.5757, 0.6168], @8192h: 0.6267
	[0.4033, 0.5183, 0.6061],	#replications: [0.3823, 0.5164, 0.6069], @8192h: 0.6025
]

# Sanity checks
#assert len(group_labels) == 5
assert len(y_values_lists) == len(group_labels)
assert all(len(y) == len(x_values) for y in y_values_lists)

# Apply graphedDatasets mask
labels_masked = subset(group_labels, graphedDatasets)
descriptions_masked = subset(group_descriptions, graphedDatasets)
y_lists_masked = [y for y, keep in zip(y_values_lists, graphedDatasets) if keep]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
for label, y in zip(labels_masked, y_lists_masked):
	if(showXaxisUniform):
		ax.plot(x_pos, y, linestyle='-', linewidth=2, label=label)
	else:
		ax.plot(x_values, y, linestyle='-', linewidth=2, label=label)

# Axes, ticks, labels
if(showXaxisUniform):
	ax.set_xticks(x_pos)
	ax.set_xticklabels([str(v) for v in x_values], rotation=0, ha='center', fontsize=10)
	ax.set_xlim(min(x_pos)-0.25, max(x_pos)+0.25)
else:
	ax.set_xticks(x_values)
	ax.set_xticklabels([str(v) for v in x_values], rotation=0, ha='center', fontsize=10)
	ax.set_xlim(min(x_values)-0.25, max(x_values)+0.25)
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.0+0.1, 0.1))
ax.grid(True, axis='y', linewidth=0.5)

ax.set_xlabel("MLP hidden layer size")
ax.set_ylabel("test accuracy (Top-1)")
ax.set_title("RPIANN performance Image Dataset (CIFAR-10) - test accuracy vs hidden layer size")

# Legend on the right
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.95), borderaxespad=0., fontsize=9)

# Right-side descriptive key box
#fig.subplots_adjust(right=0.8)
#props = dict(boxstyle='round', facecolor='white', alpha=0.8)
#ax.text(1.02, 0.5, "\n".join(descriptions_masked), transform=ax.transAxes,
#	fontsize=10, va='center', bbox=props)

plt.tight_layout()
plt.show()
