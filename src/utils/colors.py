"""Deep-sea blue theme class-color palette and auto-assignment.

类别调色板：深海蓝玻璃主题配套的 20 色高饱和亮色系，在深蓝画布上
对比度 ≥3:1，覆盖 blue/cyan/violet/green/amber/pink/red 等色相。
变量名保留 CATPPUCCIN_PALETTE 以兼容旧引用，内容已替换为主题色系。
"""

# 20 distinct colors for the deep-sea blue glass theme
CATPPUCCIN_PALETTE = [
    "#60A5FA",  # blue
    "#22D3EE",  # cyan
    "#A78BFA",  # violet
    "#34D399",  # green
    "#FBBF24",  # amber
    "#F472B6",  # pink
    "#F87171",  # red
    "#38BDF8",  # sky
    "#A5B4FC",  # indigo
    "#86EFAC",  # light-green
    "#FDE68A",  # yellow
    "#F0ABFC",  # fuchsia
    "#2DD4BF",  # teal
    "#FB923C",  # orange
    "#E879F9",  # purple
    "#7DD3FC",  # light-blue
    "#C4B5FD",  # lavender
    "#FCA5A5",  # light-red
    "#5EEAD4",  # light-teal
    "#FDBA74",  # light-orange
]


def assign_color(index: int) -> str:
    """Assign a color from the palette by index, wrapping around."""
    return CATPPUCCIN_PALETTE[index % len(CATPPUCCIN_PALETTE)]
