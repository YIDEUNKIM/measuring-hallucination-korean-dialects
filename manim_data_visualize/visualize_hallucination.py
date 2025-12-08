from manim import *
import pandas as pd
import numpy as np
import os
import manimpango
# matplotlib not needed for bubbles anymore, but keeping imports doesn't hurt.
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm

# --- Font Registration ---
try:
    # Register the downloaded font
    # manimpango.register_font returns a boolean (True/False) indicating success
    is_registered = manimpango.register_font("NanumMyeongjo-Regular.ttf")
    if is_registered:
         FONT_FAMILY = "NanumMyeongjo" 
         print(f"Font registered successfully. Using: {FONT_FAMILY}")
    else:
         print("Font registration returned False.")
         FONT_FAMILY = "sans-serif"
except Exception as e:
    print(f"Font registration warning: {e}")
    FONT_FAMILY = "sans-serif" # Fallback

# --- Configurations ---
TRUTHFULQA_TOTAL = 603
MEDNLI_TOTAL = 1372

# Province coordinates
PROVINCE_POSITIONS = {
    "표준": UP * 1.2 + LEFT * 0.5,   
    "충청도": UP * 0.2 + LEFT * 0.5, 
    "경상도": DOWN * 0.8 + RIGHT * 0.8, 
    "전라도": DOWN * 1.2 + LEFT * 0.8,  
    "제주도": DOWN * 2.5 + LEFT * 0.5   
}

# --- Data Loading Helper ---
def load_hallucination_data(csv_path, total_questions, is_accuracy=True):
    print(f"DEBUG: Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"DEBUG: First 2 rows of {csv_path}:")
    print(df.head(2))
    
    # Normalize columns: remove whitespace
    df.columns = [c.strip() for c in df.columns]
    
    region_col = df.columns[0]
    data = {}
    
    # Map possible CSV column names to Internal Model Names
    model_map = {
        "Gemini 3": "Gemini 3",
        "Claude 4.5 sonnet": "Claude 4.5 sonnet",
        "GPT 5": "GPT 5.1",   # Map Old -> New
        "GPT 5.1": "GPT 5.1"  # Map New -> New
    }
    
    for _, row in df.iterrows():
        region = row[region_col]
        if region not in PROVINCE_POSITIONS:
            continue 
        data[region] = {}
        
        for csv_col, internal_name in model_map.items():
            if csv_col in df.columns:
                val = row[csv_col]
                # cleanup string if needed
                if isinstance(val, str):
                    val = val.replace(",", "")
                score = float(val)
                
                if is_accuracy:
                    # Transform Accuracy -> Hallucination
                    hallucination = total_questions - score
                else:
                    # Already Hallucination/Error metric
                    hallucination = score
                    
                data[region][internal_name] = hallucination
                
    return data

# --- Scene 1 & 2: Bubble Map (Refined) ---
# "Semi-transparent red circles sized by data value"
class BubbleMapScene(Scene):
    def construct_scene(self, dataset_name, total_questions, csv_file, bubble_color=RED, is_accuracy=True, explanation_str=None):
        # 1. Load Data
        data = load_hallucination_data(csv_file, total_questions, is_accuracy=is_accuracy)

        # 2. Draw Map
        map_svg = SVGMobject("south_korea.svg")
        map_svg.set_fill(color="#222222", opacity=1.0)
        map_svg.set_stroke(color=GRAY, width=1)
        map_svg.height = 6
        map_svg.move_to(ORIGIN)
        
        # Use registered serif font
        title = Text(f"{dataset_name} Hallucination Bubble Chart", font=FONT_FAMILY, font_size=36)
        title.to_edge(UP)
        
        anim_group = [DrawBorderThenFill(map_svg), Write(title)]
        
        # Scoring Explanation Text (Middle Left) - Optional
        if explanation_str:
            explanation_text = Text(explanation_str, font=FONT_FAMILY, font_size=24, color=GRAY_A)
            explanation_text.to_edge(LEFT).shift(UP * 0.5 + RIGHT * 0.5)
            anim_group.append(FadeIn(explanation_text))
            
        # Min-Max Explanation Text (Bottom Left)
        scale_text = Text("Bubble Size Scaled by Min-Max", font=FONT_FAMILY, font_size=16, color=GRAY_A)
        scale_text.to_corner(DL).shift(UP * 0.5 + RIGHT * 0.5)
        anim_group.append(FadeIn(scale_text))
        
        self.play(*anim_group, run_time=2)
        
        regions_order = ["표준", "충청도", "전라도", "경상도", "제주도"]
        
        # Calculate Average for bubble size
        avg_data = {}
        max_val = 0
        min_val = float('inf')
        
        for reg in regions_order:
            if reg in data:
                vals = [data[reg][m] for m in ["Gemini 3", "GPT 5.1", "Claude 4.5 sonnet"] if m in data[reg]]
                avg_val = np.mean(vals) if vals else 0
                avg_data[reg] = avg_val
                if avg_val > max_val: max_val = avg_val
                if avg_val < min_val: min_val = avg_val
        
        bubbles = VGroup()
        labels = VGroup()
        
        # Calculate min/max for scaling
        all_vals = list(avg_data.values())
        if not all_vals:
            max_val = 1
            min_val = 0
        else:
            max_val = max(all_vals)
            min_val = min(all_vals)
            
        # Avoid division by zero if all values are same
        val_range = max_val - min_val
        if val_range == 0: val_range = 1.0

        for region in regions_order:
            if region not in avg_data:
                continue
            
            val = avg_data[region]
            
            # Bubble Size Logic (Min-Max Scaling for Contrast)
            # Map [min_val, max_val] -> [0.4, 0.95]
            # This emphasizes height differences even if absolute variance is low relative to total magnitude
            
            normalized = (val - min_val) / val_range
            radius = 0.4 + (normalized * 0.55) # 0.4 to 0.95
                
            pos = PROVINCE_POSITIONS[region]
            
            # Use specified bubble_color
            circle = Circle(radius=radius, color=bubble_color, fill_color=bubble_color, fill_opacity=0.4)
            circle.move_to(pos)
            
            label_txt = f"{region}\n{val:.0f}"
            label = Text(label_txt, font=FONT_FAMILY, font_size=20, color=WHITE)
            label.move_to(pos)
            
            self.play(FadeIn(circle), Write(label), run_time=1.0)
            
            bubbles.add(circle)
            labels.add(label)
            
        self.wait(10)
        
        # Cleanup
        cleanup_group = [FadeOut(bubbles), FadeOut(labels), FadeOut(map_svg), FadeOut(title), FadeOut(scale_text)]
        if explanation_str:
            cleanup_group.append(FadeOut(explanation_text))
        self.play(*cleanup_group)

class Scene1_TruthfulQA_Bubbles(BubbleMapScene):
    def construct(self):
        # TruthfulQA uses default RED (or standard)
        # TruthfulQA CSV is Accuracy -> Needs Inversion (is_accuracy=True)
        exp = (
            f"{TRUTHFULQA_TOTAL}개 문제 중\n"
            "정답(+1) + 모름(0) + 오답(-1)\n"
            "점수 합산\n"
            "(총점 - 점수 = 환각 수)"
        )
        self.construct_scene("TruthfulQA", TRUTHFULQA_TOTAL, "csv_data/TruthfulQA_Hallucination.csv", bubble_color=RED, is_accuracy=True, explanation_str=exp)

class Scene2_MedNLI_Bubbles(BubbleMapScene):
    def construct(self):
        # MedNLI now treated as Accuracy-like (High Score = Good) -> Invert for Hallucination
        exp = (
            f"{MEDNLI_TOTAL}개 문제 중\n"
            "정답(+1) + 모름(0) + 오답(-1)\n"
            "점수 합산\n"
            "(총점 - 점수 = 환각 수)"
        )
        self.construct_scene("MedNLI", MEDNLI_TOTAL, "csv_data/Mednli_Hallucination.csv", bubble_color=BLUE, is_accuracy=True, explanation_str=exp)


# --- Scene 3 & 4: Radar Chart (Refined) ---
class RadarChartScene(Scene):
    def construct_scene(self, dataset_name, total_questions, csv_file, is_accuracy=True):
        # 1. Load Data
        data = load_hallucination_data(csv_file, total_questions, is_accuracy=is_accuracy)
        
        if dataset_name == "TruthfulQA":
            # Debug removed
            pass
        
        regions = ["표준", "충청도", "제주도", "전라도", "경상도"] 
        
        # Calculate Min/Max for Scaling
        all_vals = []
        for r in regions:
            for m in ["Gemini 3", "GPT 5.1", "Claude 4.5 sonnet"]:
                all_vals.append(data[r][m])
        
        if not all_vals:
            max_val_data = 1
            min_val_data = 0
        else:
            max_val_data = max(all_vals)
            min_val_data = min(all_vals)
            
        val_range = max_val_data - min_val_data
        if val_range == 0: val_range = 1.0
        
        radius = 3.0
        center = DOWN * 0.5 
        
        axes = VGroup()
        axis_labels = VGroup()
        angles = np.linspace(90, 90-360, len(regions), endpoint=False) * DEGREES
        
        web = VGroup()
        grid_labels = VGroup()
        
        # Grid lines and Labels
        # r=0 (Center) Label
        center_val = min_val_data
        center_lbl = Text(f"{int(center_val)}", font=FONT_FAMILY, font_size=16, color=GRAY)
        center_lbl.add_background_rectangle(opacity=0.6, buff=0.05)
        # Position slightly offset to not be covered by lines completely
        center_lbl.move_to(center + DL * 0.2) 
        grid_labels.add(center_lbl)

        for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
            points = [center + np.array([np.cos(a), np.sin(a), 0]) * radius * r for a in angles]
            web.add(Polygon(*points, color=GRAY, stroke_opacity=0.5))
            
            # Value Label
            val = min_val_data + (val_range * r)
            label_pos = center + np.array([0, radius * r, 0]) + RIGHT * 0.2
            lbl = Text(f"{int(val)}", font=FONT_FAMILY, font_size=16, color=GRAY).move_to(label_pos)
            lbl.add_background_rectangle(opacity=0.6, buff=0.05)
            grid_labels.add(lbl)
        
        for i, angle in enumerate(angles):
            end_point = center + np.array([np.cos(angle), np.sin(angle), 0]) * radius
            line = Line(center, end_point, color=GRAY)
            axes.add(line)
            label_pos = end_point * 1.1
            
            # Manual Adjustment for overlap
            if regions[i] == "충청도":
                label_pos += RIGHT * 0.25
            elif regions[i] == "경상도":
                label_pos += LEFT * 0.25
                
            label = Text(regions[i], font=FONT_FAMILY, font_size=24).move_to(label_pos)
            axis_labels.add(label)

        title = Text(f"{dataset_name} Hallucination Radar Chart", font=FONT_FAMILY, font_size=36).to_edge(UP)
        
        # Explanation Text (Dynamic)
        scale_text = Text(
            f"Min-Max Scaled\nCenter: {int(min_val_data)} (Min)\nEdge: {int(max_val_data)} (Max)", 
            font=FONT_FAMILY, font_size=16, color=GRAY_A
        )
        scale_text.to_corner(DL).shift(UP * 0.5 + RIGHT * 0.5)

        self.play(Create(web), Create(axes), Write(axis_labels), Write(title), FadeIn(grid_labels), Write(scale_text), run_time=2)
        
        models = [
            {"name": "GPT 5.1", "color": GREEN},
            {"name": "Claude 4.5 sonnet", "color": ORANGE},
            {"name": "Gemini 3", "color": BLUE}
        ]
        
        legend = VGroup()
        chart_elements = VGroup()
        
        # "Model names in middle left"
        legend_start_pos = LEFT * 6.0 + UP * 1.0 
        
        for i, model_info in enumerate(models):
            m_name = model_info["name"]
            color = model_info["color"]
            points = []
            for j, region in enumerate(regions):
                val = data[region][m_name]
                
                # Normalize: (val - min) / range
                normalized = (val - min_val_data) / val_range
                # Visual Radius: normalized * max_radius
                # If we want 0 to be center, ensure 0 maps to 0. But here min maps to 0?
                # Usually Radar Min is center.
                # Avoid exact 0 for visibility? No, let's stick to true Min-Max.
                r_norm = normalized * radius
                
                angle = angles[j]
                pt = center + np.array([np.cos(angle), np.sin(angle), 0]) * r_norm
                points.append(pt)
            
            poly = Polygon(*points, color=color, stroke_width=4)
            leg_dot = Dot(color=color)
            leg_txt = Text(m_name, font=FONT_FAMILY, font_size=20, color=color)
            leg_item = VGroup(leg_dot, leg_txt).arrange(RIGHT)
            
            # Position: Middle Left, stacked
            leg_item.move_to(LEFT * 5.5 + UP * (1.0 - i * 0.5))
            
            legend.add(leg_item)
            chart_elements.add(poly)
            
            self.play(Create(poly), FadeIn(leg_item), run_time=4.0)
            
            self.wait(10)
        
        # Cleanup for Combined Scene
        self.play(FadeOut(web), FadeOut(axes), FadeOut(axis_labels), FadeOut(title), FadeOut(legend), FadeOut(chart_elements), FadeOut(grid_labels), FadeOut(scale_text))

class Scene3_TruthfulQA_Radar(RadarChartScene):
    def construct(self):
        self.construct_scene("TruthfulQA", TRUTHFULQA_TOTAL, "csv_data/TruthfulQA_Hallucination.csv", is_accuracy=True)

class Scene4_MedNLI_Radar(RadarChartScene):
    def construct(self):
        self.construct_scene("MedNLI", MEDNLI_TOTAL, "csv_data/Mednli_Hallucination.csv", is_accuracy=True)

# --- Combined Scene ---
class FullPresentation(BubbleMapScene, RadarChartScene): 
    def construct(self):
        # Scene 1: TruthfulQA Map
        exp_truth = (
            f"{TRUTHFULQA_TOTAL}개 문제 중\n"
            "정답(+1) + 모름(0) + 오답(-1)\n"
            "점수 합산\n"
            "(총점 - 점수 = 환각 수)"
        )
        self.construct_scene("TruthfulQA", TRUTHFULQA_TOTAL, "csv_data/TruthfulQA_Hallucination.csv", bubble_color=RED, is_accuracy=True, explanation_str=exp_truth)
        self.wait(1)
        
        # Scene 2: MedNLI Map (Blue/Purple)
        exp_med = (
            f"{MEDNLI_TOTAL}개 문제 중\n"
            "정답(+1) + 모름(0) + 오답(-1)\n"
            "점수 합산\n"
            "(총점 - 점수 = 환각 수)"
        )
        self.construct_scene("MedNLI", MEDNLI_TOTAL, "csv_data/Mednli_Hallucination.csv", bubble_color=BLUE, is_accuracy=True, explanation_str=exp_med)
        self.wait(1)
        
        # Scene 3: TruthfulQA Radar
        RadarChartScene.construct_scene(self, "TruthfulQA", TRUTHFULQA_TOTAL, "csv_data/TruthfulQA_Hallucination.csv", is_accuracy=True)
        self.wait(1)

        # Scene 4: MedNLI Radar
        RadarChartScene.construct_scene(self, "MedNLI", MEDNLI_TOTAL, "csv_data/Mednli_Hallucination.csv", is_accuracy=True)
        self.wait(1)

        # Scene 5: TruthfulQA Scatter
        ScatterScene.construct_scene(self, "TruthfulQA", TRUTHFULQA_TOTAL, "csv_data/TruthfulQA_Accuracy.csv")
        self.wait(1)

        # Scene 6: MedNLI Scatter
        ScatterScene.construct_scene(self, "MedNLI", MEDNLI_TOTAL, "csv_data/Mednli_Accuracy.csv")
        self.wait(1)


class ScatterScene(Scene):
    def construct_scene(self, dataset_name, max_val_reference, accuracy_csv):
        # 1. Setup Data
        df = pd.read_csv(accuracy_csv)
        df.columns = [c.strip() for c in df.columns]
        
        # Handle column name mismatch (GPT 5 in CSV -> GPT 5.1 in code)
        if "GPT 5" in df.columns:
            df.rename(columns={"GPT 5": "GPT 5.1"}, inplace=True)
        
        models = [
            {"name": "GPT 5.1", "col": "GPT 5.1", "color": GREEN},
            {"name": "Claude 4.5", "col": "Claude 4.5 Sonnet", "color": ORANGE}, 
            {"name": "Gemini 3", "col": "Gemini 3", "color": BLUE},
        ]
        
        # Region Config: Sequence and Conceptual Offsets
        regions_seq = [
            {"name": "표준", "label": "표준어", "offset_factor": 0.05, "color": WHITE},
            {"name": "충청도", "label": "충청도", "offset_factor": -0.05, "color": YELLOW},
            {"name": "전라도", "label": "전라도", "offset_factor": -0.08, "color": TEAL},
            {"name": "경상도", "label": "경상도", "offset_factor": -0.10, "color": ORANGE},
            {"name": "제주도", "label": "제주도", "offset_factor": -0.20, "color": RED},
        ]

        # Calculate dynamic ranges
        all_x = []
        all_y = []

        def calc_y(x, offset_factor, ref_val):
            val = x + (ref_val * offset_factor)
            return val

        # Pre-scan for ranges
        for _, row in df.iterrows():
            reg = row["Region"]
            offset = 0
            for r in regions_seq:
                if r["name"] == reg:
                    offset = r["offset_factor"]
                    break
            
            for m in models:
                if m["col"] in row:
                    val_str = str(row[m["col"]]).replace(",", "")
                    try:
                        x_val = float(val_str)
                        y_val = calc_y(x_val, offset, max_val_reference)
                        all_x.append(x_val)
                        all_y.append(y_val)
                    except:
                        pass
        
        # Determine Min/Max for Scaling
        if not all_x: all_x = [0, 1]
        if not all_y: all_y = [0, 1]

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # Ranges
        x_range_val = x_max - x_min
        if x_range_val == 0: x_range_val = 1.0
        y_range_val = y_max - y_min
        if y_range_val == 0: y_range_val = 1.0

        # Axes Configuration (Normalized 0 to 1)
        # We map data min->0, max->1
        ax = Axes(
            x_range=[0, 1.05, 0.2], # Slightly over 1 for tip
            y_range=[0, 1.05, 0.2],
            x_length=6.0,
            y_length=6.0,
            axis_config={"include_numbers": False, "font_size": 16, "tip_shape": StealthTip},
        ).move_to(ORIGIN).shift(LEFT * 0.5)
        
        # Manual Labels (Mapped back to original values)
        axis_labels_group = VGroup()
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        for t in ticks:
            # X Axis
            val_x = x_min + (x_range_val * t)
            label_x = Text(f"{int(val_x)}", font=FONT_FAMILY, font_size=14, color=GRAY)
            # Position relative to axis tick
            label_x.next_to(ax.c2p(t, 0), DOWN, buff=0.3)
            axis_labels_group.add(label_x)
            
            # Y Axis
            val_y = y_min + (y_range_val * t)
            label_y = Text(f"{int(val_y)}", font=FONT_FAMILY, font_size=14, color=GRAY)
            label_y.next_to(ax.c2p(0, t), LEFT, buff=0.3)
            axis_labels_group.add(label_y)

        # Axis Titles (Korean/English Mix as requested, Position Adjusted)
        x_label = Text("Accuracy Score (정확도)", font=FONT_FAMILY, font_size=18).next_to(ax.x_axis, DOWN, buff=0.4)
        y_label = Text("Hallucination Score (신뢰성)", font=FONT_FAMILY, font_size=18).rotate(90 * DEGREES).next_to(ax.y_axis, LEFT, buff=0.8)
        
        # Diagonal (0,0 to 1,1 in normalized space)
        diag_line = DashedLine(
            start=ax.c2p(0, 0),
            end=ax.c2p(1, 1),
            color=GRAY
        )
        
        # Zones
        # Reliable Check: Assuming Top Left as before
        reliable_zone = Polygon(
            ax.c2p(0, 0),
            ax.c2p(0, 1),
            ax.c2p(1, 1),
            color=GREEN, fill_color=GREEN, fill_opacity=0.1, stroke_opacity=0
        )
        
        overconfident_zone = Polygon(
            ax.c2p(0, 0),
            ax.c2p(1, 0),
            ax.c2p(1, 1),
            color=RED, fill_color=RED, fill_opacity=0.1, stroke_opacity=0
        )
        
        # Zone Labels (Korean)
        reliable_text = Text("신뢰 구간\n(높은 신뢰도)", font=FONT_FAMILY, font_size=16, color=GREEN).move_to(
            ax.c2p(0.25, 0.75) 
        )
        overconfident_text = Text("과신 구간\n(낮은 신뢰도)", font=FONT_FAMILY, font_size=16, color=RED).move_to(
            ax.c2p(0.75, 0.25)
        )
        
        title = Text(f"{dataset_name} Capability vs Attitude", font=FONT_FAMILY, font_size=32).to_edge(UP)

        # Explanation Text
        zoom_text = Text("데이터 범위 기반\nMin-Max 스케일링", font=FONT_FAMILY, font_size=16, color=GRAY_A)
        zoom_text.to_corner(DL).shift(UP * 0.5 + RIGHT * 0.2)

        # Setup Animation
        self.play(Write(title))
        self.play(Create(ax), Write(x_label), Write(y_label), Write(axis_labels_group), Write(zoom_text))
        self.play(FadeIn(reliable_zone), FadeIn(overconfident_zone), Create(diag_line))
        self.play(Write(reliable_text), Write(overconfident_text))
        
        # Legend (Right side)
        legend = VGroup()
        for m in models:
            leg_dot = Dot(color=m["color"])
            leg_txt = Text(m["name"], font=FONT_FAMILY, font_size=18)
            item = VGroup(leg_dot, leg_txt).arrange(RIGHT)
            legend.add(item)
        legend.arrange(DOWN, aligned_edge=LEFT).next_to(ax, RIGHT, buff=0.5).shift(UP * 0.5)
        self.play(FadeIn(legend))

        def get_data_point(region_name, model_col, offset_factor):
            row = df[df["Region"] == region_name]
            if row.empty:
                return 0, 0
            
            val_str = str(row[model_col].values[0]).replace(",", "")
            x_val = float(val_str)
            y_val = calc_y(x_val, offset_factor, max_val_reference)
            
            # Normalize
            norm_x = (x_val - x_min) / x_range_val
            norm_y = (y_val - y_min) / y_range_val
            
            return norm_x, norm_y

        # 3. Sequential Animation
        
        # Tracking Dots (Active)
        active_dots = VGroup()
        for i, m in enumerate(models):
            d = Dot(color=m["color"], radius=0.12)
            active_dots.add(d)
        
        state_label = Text("Initializing...", font=FONT_FAMILY, font_size=24, color=WHITE)
        
        final_avg_dots = VGroup()
        final_region_labels = VGroup()
        
        for i, r_info in enumerate(regions_seq):
            r_name = r_info["name"]
            r_label_txt = r_info["label"]
            offset = r_info["offset_factor"]
            r_color = r_info["color"]
            
            new_dot_positions = []
            avg_pos = np.array([0., 0., 0.])
            
            for m_idx, m in enumerate(models):
                x, y = get_data_point(r_name, m["col"], offset)
                pos = ax.c2p(x, y)
                new_dot_positions.append(pos)
                avg_pos += pos
                
            avg_pos /= len(models)
            
            avg_dot = Dot(avg_pos, color=r_color, radius=0.15)
            final_avg_dots.add(avg_dot)
            
            l_dot = Dot(color=r_color)
            l_txt = Text(r_name, font=FONT_FAMILY, font_size=18)
            l_item = VGroup(l_dot, l_txt).arrange(RIGHT)
            final_region_labels.add(l_item)

            new_label = Text(r_label_txt, font=FONT_FAMILY, font_size=24).move_to(avg_pos + UP * 0.6)
            if new_label.get_top()[1] > 3.5:
                new_label.next_to(active_dots, DOWN)
            
            if i == 0:
                for idx, dot in enumerate(active_dots):
                    dot.move_to(new_dot_positions[idx])
                state_label = new_label
                self.play(FadeIn(active_dots), Write(state_label))
            else:
                anims = [Transform(state_label, new_label)]
                for idx, dot in enumerate(active_dots):
                    anims.append(dot.animate.move_to(new_dot_positions[idx]))
                self.play(*anims, run_time=1.5)
            
            self.wait(1)
            
        self.play(FadeOut(active_dots), FadeOut(state_label), FadeOut(legend))
        final_region_labels.arrange(DOWN, aligned_edge=LEFT).move_to(legend.get_center())
        summary = Text("지역별 평균", font=FONT_FAMILY, font_size=24).to_edge(UP).shift(DOWN * 1.0)
        self.play(FadeIn(final_avg_dots), FadeIn(final_region_labels), Write(summary))
        self.wait(5)
        self.play(FadeOut(Group(*self.mobjects)))
