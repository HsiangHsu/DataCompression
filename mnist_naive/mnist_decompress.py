i = 5
retcon = [[] for i in range(28)]
while i < len(diff_string):
    row = -1
    if int(diff_string[i:i+5], base=2) == 30:
        i += 5
        row = int(diff_string[i:i+5], base=2)
        i += 5
        raw_row = []
        while int(diff_string[i:i+5], base=2) != 30:
            raw_row.append(int(diff_string[i:i+5], base=2))
            i += 5
            if i == len(diff_string):
                break
        retcon[row] = raw_row

decomp = [[] for i in range(28)]
for i in range(len(decomp)):
    if retcon[i]:
        j = 0
        while j < len(retcon[i]):
            decomp[i] += list(range(retcon[i][j], retcon[i][j+1]))
            j += 2

restored = averages[label].copy()
for i in range(28):
    for j in range(28):
        if j in decomp[i]:
            if restored[i][j] == 0:
                restored[i][j] = 1
            else:
                restored[i][j] = 0

fig = plt.figure()
ax1 = fig.add_subplot(141)
ax1.imshow(np.array(uncompressed[idx]), cmap="gray_r")
ax2 = fig.add_subplot(142)
ax2.imshow(np.array(averages[label]), cmap="gray_r")
ax3 = fig.add_subplot(143)
ax3.imshow(np.array(diff), cmap="gray_r")
ax4 = fig.add_subplot(144)
ax4.imshow(np.array(restored), cmap="gray_r")
plt.show()
