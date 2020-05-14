import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1,1) 

plt.ylabel('Size (MB)')
plt.xlabel('')
x = np.arange(0, 5, 1)
x_ticks_labels = ['Zipped random mkv','Random order webm','Zipped 100-NN MST mkv','100-NN MST webm','Zipped 100-NN MST webm']

'''
av1_png_gop = 10**-6 * np.array([2289261.0, 1512023, 2281912, 1424564, 1326199])
av1_png_no_gop = 10**-6 * np.array([2297438.0, 1520987, 2289326, 1432102, 1336792])
av1_jpg_gop = 10**-6 * np.array([4292363.0, 1527686, 4281270, 1437416, 1338039])
av1_jpg_no_gop = 10**-6 * np.array([2452290.0, 1681760, 2446362, 1590061, 1498073])
ax.plot(x, av1_png_gop, marker='.', label='AV1 PNG GoP 191')
ax.plot(x, av1_png_no_gop, marker='.', label='AV1 PNG GoP default (128)')
ax.plot(x, av1_jpg_gop, marker='.', label='AV1 JPG GoP 175')
ax.plot(x, av1_jpg_no_gop, marker='.', label='AV1 JPG GoP default (128)')


vp8_png_gop = 10**-6 * np.array([2292117.0, 1647541, 2284469, 1490951, 1400487])
vp8_png_no_gop = 10**-6 * np.array([2287922.0, 1645538, 2280502, 1503724, 1414302])
vp8_jpg_gop = 10**-6 * np.array([4296227.0, 1655254, 4286445, 1507320, 1418003])
vp8_jpg_no_gop = 10**-6 * np.array([4281378.0, 1646979, 4271919, 1507796, 1418317])
ax.plot(vp8_png_gop, marker='.', label='VP8 PNG GoP 193')
ax.plot(vp8_png_no_gop, marker='.', label='VP8 PNG GoP default (128)')
ax.plot(vp8_jpg_gop, marker='.', label='VP8 JPG GoP 193')
ax.plot(vp8_jpg_no_gop, marker='.', label='VP8 JPG GoP default (128)')
'''
vp9_png_gop = 10**-6 * np.array([2296368.0, 1632666, 2287496, 1520792, 1413610])
vp9_png_no_gop = 10**-6 * np.array([2290476.0, 1629015, 2282641, 1520870, 1415457])
vp9_jpg_gop = 10**-6 * np.array([4294328.0, 1630208, 4284763, 1525713, 1418520])
vp9_jpg_no_gop = 10**-6 * np.array([4301411.0, 1633648, 4292783, 1530203, 1424299])
ax.plot(vp9_png_gop, marker='.', label='VP9 PNG GoP 165')
ax.plot(vp9_png_no_gop, marker='.', label='VP9 JPG GoP default (128)')
ax.plot(vp9_jpg_gop, marker='.', label='VP9 JPG GoP 155')
ax.plot(vp9_jpg_no_gop, marker='.', label='VP9 JPG GoP default (128)')

for a in iter([vp9_png_gop, vp9_png_no_gop, vp9_jpg_gop, vp9_jpg_no_gop]):#vp8_png_gop, vp8_png_no_gop, vp8_jpg_gop, vp8_jpg_no_gop, av1_png_gop, av1_png_no_gop, av1_jpg_gop, av1_jpg_no_gop]):
	for xy in zip(x, a):
		ax.annotate("{:.4f} MB".format(xy[1]), xy, textcoords='data', ha='left')

# Set number of ticks for x-axis
ax.set_xticks(x)
# Set ticks labels for x-axis
ax.set_xticklabels(x_ticks_labels)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right",va="center",
		 rotation_mode="anchor", wrap=True)

ax.legend()

plt.show()