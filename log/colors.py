from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
for color in colors:
	colors[color] = mcolors.to_rgba(color)[:3]

color_id2label = { }
for i in range(len(colors)):
	color_id2label[i] = list(colors.keys())[i]