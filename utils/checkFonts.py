import matplotlib.font_manager as font_manager

# 获取系统中所有可用字体
available_fonts = sorted([f.name for f in font_manager.fontManager.ttflist])

# 打印字体名称
for font in available_fonts:
    print(font)
