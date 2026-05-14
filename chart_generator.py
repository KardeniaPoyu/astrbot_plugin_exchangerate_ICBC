import os
import tempfile
from datetime import datetime

import matplotlib
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

try:
    from astrbot.api import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ==========================================
# 现代化极简主题配置
# ==========================================
THEMES = {
    "light": {
        "bg_color":        "#FFFFFF",   # 纯净白底
        "card_color":      "#FFFFFF",
        "text_color":      "#111827",   # 接近纯黑的主标题
        "text_secondary":  "#6B7280",   # 柔和灰色的坐标轴标签
        "grid_color":      "#F3F4F6",   # 极浅的网格线
        "border_color":    "#E5E7EB",   # 底部轴线颜色
        "buy_color":       "#3B82F6",   # 清晰锐利的蓝色（结汇价）
        "sell_color":      "#F59E0B",   # 温暖的琥珀色（购汇价）
        "color_up":        "#10B981",   # 涨（翡翠绿）
        "color_down":      "#EF4444",   # 跌（玫瑰红）
        "anno_bg":         "#F9FAFB",   # 标注的微灰背景
    },
    "dark": {
        "bg_color":        "#111827",   # 深邃黑底
        "card_color":      "#111827",
        "text_color":      "#F9FAFB",
        "text_secondary":  "#9CA3AF",
        "grid_color":      "#1F2937",
        "border_color":    "#374151",
        "buy_color":       "#60A5FA",   # 亮蓝色
        "sell_color":      "#FBBF24",   # 亮琥珀色
        "color_up":        "#34D399",
        "color_down":      "#F87171",
        "anno_bg":         "#1F2937",
    }
}

class ExchangeRateChartGenerator:
    _cached_font_prop = None

    @classmethod
    def _get_font_prop(cls) -> fm.FontProperties:
        """获取可用的中文字体 FontProperties，跨平台兼容。"""
        if cls._cached_font_prop is not None:
            return cls._cached_font_prop

        chinese_fonts = [
            "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB",
            "SimHei", "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC"
        ]
        for name in chinese_fonts:
            try:
                path = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
                if path and os.path.exists(path) and not path.endswith("DejaVuSans.ttf"):
                    cls._cached_font_prop = fm.FontProperties(fname=path)
                    logger.info(f"汇率图表使用中文字体: {path}")
                    return cls._cached_font_prop
            except Exception:
                continue

        font_keywords = ["msyh", "yahei", "pingfang", "notosanscjk", "notosanssc", "simhei"]
        try:
            for f in fm.findSystemFonts():
                fname_lower = os.path.basename(f).lower()
                if any(k in fname_lower for k in font_keywords):
                    cls._cached_font_prop = fm.FontProperties(fname=f)
                    logger.info(f"汇率图表使用中文字体: {f}")
                    return cls._cached_font_prop
        except Exception:
            pass

        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        fonts_dir = os.path.join(plugin_dir, "fonts")
        if os.path.isdir(fonts_dir):
            for f in os.listdir(fonts_dir):
                if f.lower().endswith((".ttf", ".otf", ".ttc")):
                    path = os.path.join(fonts_dir, f)
                    try:
                        cls._cached_font_prop = fm.FontProperties(fname=path)
                        return cls._cached_font_prop
                    except Exception:
                        continue

        logger.warning("未找到可用的中文字体，图表中文可能显示异常。")
        cls._cached_font_prop = fm.FontProperties()
        return cls._cached_font_prop

    @classmethod
    def _parse_records(cls, records: list) -> tuple:
        times, buy_prices, sell_prices = [], [], []
        for r in records:
            try:
                t = datetime.strptime(r["time"], "%Y-%m-%d %H:%M")
                times.append(t)
                buy_prices.append(float(r.get("buy", 0)))
                sell_prices.append(float(r.get("sell", 0)))
            except (ValueError, KeyError):
                continue
        return times, buy_prices, sell_prices

    @classmethod
    def _draw_series(cls, ax, times, buy_prices, sell_prices, theme: dict):
        has_buy = any(p > 0 for p in buy_prices)
        has_sell = any(p > 0 for p in sell_prices)

        all_valid = [p for p in (buy_prices + sell_prices) if p > 0]
        if all_valid:
            price_range = max(all_valid) - min(all_valid)
            fill_bottom = min(all_valid) - price_range * 0.1
        else:
            fill_bottom = 0

        for prices, color, label, has_data in [
            (buy_prices, theme["buy_color"], "结汇价 (Buy)", has_buy),
            (sell_prices, theme["sell_color"], "购汇价 (Sell)", has_sell),
        ]:
            if not has_data or not prices:
                continue

            # ==========================================
            # 优雅的渐变填充 (干净、锐利)
            # ==========================================
            steps = 30
            prices_arr = np.array(prices)
            fill_arr = np.full_like(prices_arr, fill_bottom)
            
            for i in range(steps):
                frac_top = 1.0 - (i / steps)
                frac_bottom = 1.0 - ((i + 1) / steps)
                
                y_top = fill_arr + (prices_arr - fill_arr) * frac_top
                y_bottom = fill_arr + (prices_arr - fill_arr) * frac_bottom
                
                # Alpha 呈二次方锐减，确保只有线条附近有明显颜色，底部极度透明
                alpha = 0.18 * (frac_top ** 2.5)
                
                ax.fill_between(
                    times, y_top, y_bottom,
                    color=color, alpha=alpha, linewidth=0, zorder=2
                )

            # ==========================================
            # 锐利的主线条
            # ==========================================
            ax.plot(
                times, prices, linewidth=2.2, label=label,
                color=color, zorder=4, solid_capstyle="round"
            )

            # ==========================================
            # 现代感环形指示点 (Latest Marker)
            # ==========================================
            lx, ly = times[-1], prices[-1]
            # 外围淡色光晕
            ax.scatter([lx], [ly], s=120, color=color, alpha=0.15, zorder=4, linewidths=0)
            # 核心双色圆环
            ax.scatter([lx], [ly], s=40, color=theme["bg_color"], zorder=5, 
                       edgecolors=color, linewidths=2.5)

    @classmethod
    def _add_annotations(cls, ax, times, buy_prices, sell_prices, theme: dict, font_dict: dict):
        has_buy = any(p > 0 for p in buy_prices)
        has_sell = any(p > 0 for p in sell_prices)

        for prices, color, has_data in [
            (buy_prices, theme["buy_color"], has_buy),
            (sell_prices, theme["sell_color"], has_sell),
        ]:
            if not has_data: continue
            valid = [(t, p) for t, p in zip(times, prices) if p > 0]
            if len(valid) < 2: continue

            valid_prices = [p for _, p in valid]
            max_val = max(valid_prices)
            min_val = min(valid_prices)
            if max_val == min_val: continue

            max_idx = valid_prices.index(max_val)
            min_idx = valid_prices.index(min_val)
            max_time = valid[max_idx][0]
            min_time = valid[min_idx][0]

            # 极简气泡样式（无箭头指示线）
            bubble_style = dict(
                boxstyle="round,pad=0.3,rounding_size=0.4",
                facecolor=theme["anno_bg"],
                edgecolor=color,
                linewidth=1,
                alpha=0.9
            )

            # 最高价
            ax.annotate(
                f"{max_val:.4f}",
                xy=(max_time, max_val),
                xytext=(0, 10),
                textcoords="offset points",
                fontproperties=font_dict["fp_small"],
                color=color,
                ha="center", va="bottom",
                bbox=bubble_style,
                zorder=6
            )

            # 最低价
            ax.annotate(
                f"{min_val:.4f}",
                xy=(min_time, min_val),
                xytext=(0, -10),
                textcoords="offset points",
                fontproperties=font_dict["fp_small"],
                color=color,
                ha="center", va="top",
                bbox=bubble_style,
                zorder=6
            )

        # 最新值 & 涨跌幅标注在右侧
        offsets = [(30, 0), (30, -25)] # 错开避免重叠
        for idx, (prices, color, has_data) in enumerate([
            (buy_prices, theme["buy_color"], has_buy),
            (sell_prices, theme["sell_color"], has_sell)
        ]):
            if not has_data or not prices: continue
            
            latest = prices[-1]
            first_val = prices[0]
            
            # 主气泡
            ax.annotate(
                f"{latest:.4f}",
                xy=(times[-1], latest),
                xytext=(20, 15 if idx==0 else -15),
                textcoords="offset points",
                fontproperties=font_dict["fp_anno"],
                color=color,
                ha="left", va="center",
                bbox=dict(
                    boxstyle="round,pad=0.4,rounding_size=0.4",
                    facecolor=theme["anno_bg"],
                    edgecolor=color,
                    linewidth=1.2,
                    alpha=0.95
                ),
                zorder=7
            )

            if first_val > 0:
                change = latest - first_val
                pct = change / first_val * 100
                c = theme["color_up"] if change >= 0 else theme["color_down"]
                sign = "+" if change >= 0 else ""
                
                # 涨跌幅紧挨气泡
                ax.annotate(
                    f"{sign}{pct:.2f}%",
                    xy=(times[-1], latest),
                    xytext=(65, 15 if idx==0 else -15),
                    textcoords="offset points",
                    fontproperties=font_dict["fp_small_bold"],
                    color=c,
                    ha="left", va="center",
                    zorder=7
                )

    @classmethod
    def _apply_theme_and_layout(cls, fig, ax, currency: str, times: list, theme: dict, font_dict: dict):
        fig.set_facecolor(theme["bg_color"])
        ax.set_facecolor(theme["card_color"])

        # ==========================================
        # 极简坐标轴：隐藏顶部、右侧、左侧轴线，仅保留底部主线
        # ==========================================
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(theme["border_color"])
        ax.spines["bottom"].set_linewidth(1.2)

        # ==========================================
        # 标题与副标题
        # ==========================================
        ax.set_title(f"{currency} Exchange Rate", fontproperties=font_dict["fp_title"], color=theme["text_color"], pad=25, loc="left", x=0)
        
        if times:
            date_range = f"{times[0].strftime('%Y/%m/%d %H:%M')} - {times[-1].strftime('%Y/%m/%d %H:%M')}"
            ax.text(0, 1.04, date_range, transform=ax.transAxes, fontproperties=font_dict["fp_subtitle"], color=theme["text_secondary"], va="bottom")

        # ==========================================
        # 网格：仅保留水平点阵网格，视觉干扰降到最低
        # ==========================================
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color=theme["grid_color"], alpha=0.9)
        ax.set_axisbelow(True)

        # ==========================================
        # 刻度及标签
        # ==========================================
        ax.tick_params(axis="both", colors=theme["text_secondary"], length=0, labelsize=9, pad=8)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_dict["fp_tick"])

        if len(times) > 1:
            span = (times[-1] - times[0]).total_seconds()
            if span > 86400 * 3:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        fig.autofmt_xdate(rotation=0, ha="center")

        # ==========================================
        # 图例：无边框浮动设计
        # ==========================================
        ax.legend(
            prop=font_dict["fp_legend"],
            facecolor=theme["bg_color"],
            edgecolor="none",
            labelcolor=theme["text_color"],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            framealpha=0.5,
            borderpad=0,
            handletextpad=0.5,
            handlelength=1.2
        )

        # 水印
        fig.text(0.95, 0.03, "Data Source: ICBC", fontproperties=font_dict["fp_watermark"], color=theme["text_secondary"], ha="right", va="bottom", alpha=0.4)

        fig.subplots_adjust(left=0.08, right=0.88, top=0.85, bottom=0.1)

    @classmethod
    def generate(cls, currency: str, records: list, theme_name: str = "light") -> str | None:
        try:
            times, buy_prices, sell_prices = cls._parse_records(records)
            if not times:
                return None

            font_prop = cls._get_font_prop()
            theme = THEMES.get(theme_name, THEMES["light"])

            def make_font(size: float, weight: str = "normal"):
                fp = font_prop.copy()
                fp.set_size(size)
                fp.set_weight(weight)
                return fp

            font_dict = {
                "fp_title": make_font(18, "bold"),
                "fp_subtitle": make_font(10),
                "fp_label": make_font(11, "bold"),
                "fp_legend": make_font(10, "bold"),
                "fp_tick": make_font(9),
                "fp_anno": make_font(10, "bold"),
                "fp_small": make_font(8.5),
                "fp_small_bold": make_font(8.5, "bold"),
                "fp_watermark": make_font(8),
            }

            with matplotlib.rc_context({"axes.unicode_minus": False}):
                fig = Figure(figsize=(12, 5.5), dpi=150)
                canvas = FigureCanvasAgg(fig)
                
                ax = fig.add_subplot(111)

                cls._draw_series(ax, times, buy_prices, sell_prices, theme)
                cls._apply_theme_and_layout(fig, ax, currency, times, theme, font_dict)
                cls._add_annotations(ax, times, buy_prices, sell_prices, theme, font_dict)

                fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="icbc_chart_")
                os.close(fd)
                canvas.print_figure(
                    tmp_path,
                    dpi=150,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                    edgecolor="none",
                )

                fig.clf()
                plt.close(fig)
                return tmp_path

        except Exception as e:
            logger.error(f"生成汇率图表失败: {e}", exc_info=True)
            return None
