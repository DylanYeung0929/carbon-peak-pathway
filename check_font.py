import matplotlib.font_manager as fm

# 设置常见的中文字体关键词
keywords = ['PingFang', 'Heiti', 'Songti', 'Kaiti', 'STHeiti', 'STSong', 'Arial Unicode MS', 'SimHei', 'FangSong']
available_fonts = set(f.name for f in fm.fontManager.ttflist)

print("你的 Matplotlib 目前认识的潜在中文字体有：")
for font in available_fonts:
    if any(kw in font for kw in keywords):
        print("-", font)