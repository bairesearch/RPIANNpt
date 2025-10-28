import matplotlib.pyplot as plt
import numpy as np

def subset(arr, subsetMask):
	arr = np.array(arr)[np.array(subsetMask)]
	return arr

showXaxisUniform = True

# groups to plot
graphedDatasets = [True, True, True, True]

x_values = [512, 2048, 8192]
if(showXaxisUniform):
	x_pos = np.arange(len(x_values))

# One label per group (these are also the legend keys lines)
group_labels = ["numberOfLayers=9", "numberOfLayers=9 trainLocal=False", "numberOfLayers=9 useRecursiveLayers", "numberOfLayers=1"]
	
# Optional long descriptions shown in the right-side key
group_descriptions = [
	#"convlay=1 mlplay=1 classLayLoss concatInp+residInp",
	#"convlay=1 mlplay=1 concatInp+residInp",
	#"convlay=1 mlplay=9 noOutProjAct sublay=2 reclay concatInp+residInp",
	#"convlay=1 mlplay=1 noOutProjAct sublay=2 concatInp+residInp",
	#"convlay=1 mlplay=9 noOutProjAct reclay",
	#"convlay=1 mlplay=9 noOutProjAct",
	#"convlay=1 mlplay=1 noOutProjAct",
	
	#"convlay=1 mlplay=9 inp+outProjActTanh reclay",
	#"convlay=1 mlplay=9 inp+outProjActTanh",
	#"convlay=1 mlplay=1 inp+outProjActTanh",
	#"convlay=1 mlplay=9 inp+outProjActTanh concatInp reclay",
	#"convlay=1 mlplay=9 inp+outProjActTanh concatInp",
	#"convlay=1 mlplay=1 inp+outProjActTanh concatInp",
	
	#"convlay=1 mlplay=9 inp+outProjActTanh concatInp initialiseYhatZero reclay sublay=2" subLayerFirstMixXembedYhatStreamsSeparately+subLayerFirstSparse",
	#"convlay=1 mlplay=9 inp+outProjActTanh concatInp initialiseYhatZero reclay sublay=2" subLayerFirstSparse",
	#"convlay=1 mlplay=9 inp+outProjActTanh concatInp initialiseYhatZero reclay sublay=2" subLayerFirstMixXembedYhatStreamsSeparately",
	#"convlay=1 mlplay=9 inp+outProjActTanh concatInp initialiseYhatZero reclay sublay=2",
	
	"convlay=1 mlplay=9 inp+outProjActTanh concatInp initialiseYhatZero",
	"convlay=1 mlplay=9 inp+outProjActTanh concatInp initialiseYhatZero trainBackprop",
	"convlay=1 mlplay=9 inp+outProjActTanh concatInp initialiseYhatZero reclay",
	"convlay=1 mlplay=1 inp+outProjActTanh concatInp",
	
	#"convlay=1 mlplay=1 inp+outProjActTanh concatInp initialiseYhatZero",
	#"convlay=1 mlplay=1 inp+outProjActTanh concatInp initialiseYhatZero repeatTrainLoopX9",
]

y_values_lists = [
	#[0.4033, 0.4976, 0.5823],
	#[0.3031, 0.4412, 0.5145],
	#[0.3311, 0.4618, 0.3354],
	#[0.3679, 0.4494, 0.5254],
	#[0.1000, 0.4004, 0.4908],
	#[0.2815, 0.4011, 0.4672],
	#[0.3029, 0.4149, 0.4932],
	
	#[0.2031, 0.3962, 0.4873],
	#[0.1961, 0.4527, 0.3554],
	#[0.2430, 0.4268, 0.4710],
	#[0.2975, 0.4399, 0.5136],
	#[0.2794, 0.4324, 0.5028],
	#[0.3041, 0.3836, 0.3781],	#replication @8192h: 0.3969
	
	#[0.3132, 0.4535, ],
	#[0.3533, 0.4531, ],
	#[0.3497, 0.4690, 0.5262],
	#[0.3362, 0.4597, ],
	
	[0.4079, 0.4224, 0.4711],	#replication1 @8192h: 0.5062 #replication2 @8192h: 0.4955
	[0.4045, 0.4765, 0.5185],	#replication1 @8192h: 0.5222
	[0.3110, 0.4590, 0.5306],	#replication1 @8192h: 0.5031 #replication2 @8192h: 0.5351
	[0.3298, 0.3616, 0.5038],
	
	#[0.3088, 0.4285, 0.4939],
	#[0.2416, 0.4317, 0.5208],
]

# Sanity checks
#assert len(group_labels) == 4
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
