from manim import *
import pandas as pd
import numpy as np
import os
import manimpango

# --- Font Registration ---
try:
    is_registered = manimpango.register_font("NanumMyeongjo-Regular.ttf")
    if is_registered:
         FONT_FAMILY = "NanumMyeongjo" 
    else:
         FONT_FAMILY = "sans-serif"
except Exception as e:
    print(f"Font registration warning: {e}")
    FONT_FAMILY = "sans-serif"

class ScatterScene(Scene):
    def construct_scene(self, dataset_name, max_val, accuracy_csv):
        # 1. Setup Data
        df = pd.read_csv(accuracy_csv)
        df.columns = [c.strip() for c in df.columns]
        
        models = [
            {"name": "GPT 5.1", "col": "GPT 5.1", "color": GREEN},
            {"name": "Claude 4.5", "col": "Claude 4.5 Sonnet", "color": ORANGE}, 
            {"name": "Gemini 3", "col": "Gemini 3", "color": BLUE},
        ]
        
        # Region Config: Sequence and Conceptual Offsets
        # Sequence: Standard -> Chungcheong -> Gyeongsang -> Jeolla -> Jeju
        # Colors for Final Scene (Distinct per region)
        regions_seq = [
            {"name": "표준", "label": "표준어", "offset_factor": 0.05, "color": WHITE},
            {"name": "충청도", "label": "충청도", "offset_factor": -0.05, "color": YELLOW},
            {"name": "전라도", "label": "전라도", "offset_factor": -0.08, "color": TEAL},
            {"name": "경상도", "label": "경상도", "offset_factor": -0.10, "color": ORANGE},
            {"name": "제주도", "label": "제주도", "offset_factor": -0.20, "color": RED},
        ]

        def get_data_point(region_name, model_col, offset_factor):
            row = df[df["Region"] == region_name]
            if row.empty:
                return 0, 0
            
            val_str = str(row[model_col].values[0]).replace(",", "")
            x_val = float(val_str)
            
            # Y = X + Offset
            offset = max_val * offset_factor
            y_val = x_val + offset
            
            # Caps
            if y_val > max_val: y_val = max_val
            if y_val < 0: y_val = 0
                
            return x_val, y_val

        # 2. Setup Axes (Centered)
        # Shift slightly LEFT to balance with Legend
        ax = Axes(
            x_range=[0, max_val, max_val // 10],
            y_range=[0, max_val, max_val // 10],
            x_length=5.5,  # Reduced from 7
            y_length=5.5,  # Reduced from 7
            axis_config={"include_numbers": False, "tip_shape": StealthTip},
        ).move_to(ORIGIN).shift(LEFT * 0.8) # Shifted more left since it's smaller
        
        x_label = Text("Accuracy Score (Capability)", font=FONT_FAMILY, font_size=18).next_to(ax.x_axis, DOWN)
        y_label = Text("Hallucination Score (Reliability)", font=FONT_FAMILY, font_size=18).rotate(90 * DEGREES).next_to(ax.y_axis, LEFT)
        
        # Diagonal
        diag_line = DashedLine(
            start=ax.c2p(0, 0),
            end=ax.c2p(max_val, max_val),
            color=GRAY
        )
        
        # Zones
        reliable_zone = Polygon(
            ax.c2p(0, 0), ax.c2p(0, max_val), ax.c2p(max_val, max_val),
            color=GREEN, fill_color=GREEN, fill_opacity=0.1, stroke_opacity=0
        )
        overconfident_zone = Polygon(
            ax.c2p(0, 0), ax.c2p(max_val, 0), ax.c2p(max_val, max_val),
            color=RED, fill_color=RED, fill_opacity=0.1, stroke_opacity=0
        )
        
        # Zone Labels (Adjusted size)
        reliable_text = Text("Reliable Zone\n(Unknown 사용)", font=FONT_FAMILY, font_size=16, color=GREEN).move_to(
            ax.c2p(max_val * 0.2, max_val * 0.85)
        )
        overconfident_text = Text("Overconfident Zone\n(모르는데 아는 척)", font=FONT_FAMILY, font_size=16, color=RED).move_to(
            ax.c2p(max_val * 0.8, max_val * 0.15)
        )

        title = Text(f"{dataset_name} Capability vs Attitude", font=FONT_FAMILY, font_size=32).to_edge(UP)

        # Setup Animation
        self.play(Write(title))
        self.play(Create(ax), Write(x_label), Write(y_label))
        self.play(FadeIn(reliable_zone), FadeIn(overconfident_zone), Create(diag_line))
        self.play(Write(reliable_text), Write(overconfident_text))
        
        # Legend (Right side)
        legend = VGroup()
        for m in models:
            leg_dot = Dot(color=m["color"])
            leg_txt = Text(m["name"], font=FONT_FAMILY, font_size=18)
            item = VGroup(leg_dot, leg_txt).arrange(RIGHT)
            legend.add(item)
        legend.arrange(DOWN, aligned_edge=LEFT).next_to(ax, RIGHT, buff=1.0).shift(UP * 0.5)
        self.play(FadeIn(legend))


        # 3. Sequential Animation
        
        # Tracking Dots (Active)
        active_dots = VGroup()
        for i, m in enumerate(models):
            # Init at Origin or start pos? Start at Standard.
            d = Dot(color=m["color"], radius=0.12) # Slightly larger
            active_dots.add(d)
        
        # State Label (Dynamic)
        state_label = Text("Initializing...", font=FONT_FAMILY, font_size=24, color=WHITE)
        
        # Store for final avg scene
        final_avg_dots = VGroup()
        final_region_labels = VGroup()
        
        # Loop regions
        for i, r_info in enumerate(regions_seq):
            r_name = r_info["name"]
            r_label_txt = r_info["label"]
            offset = r_info["offset_factor"]
            r_color = r_info["color"]
            
            # Calculate new positions
            new_dot_positions = []
            avg_pos = np.array([0., 0., 0.])
            
            for m_idx, m in enumerate(models):
                x, y = get_data_point(r_name, m["col"], offset)
                pos = ax.c2p(x, y)
                new_dot_positions.append(pos)
                avg_pos += pos
                
            avg_pos /= len(models)
            
            # Create Average Dot for Final Scene
            avg_dot = Dot(avg_pos, color=r_color, radius=0.15)
            # Add simple text label next to the dot? Or rely on legend?
            # User wants "Color by region". A legend is safer.
            final_avg_dots.add(avg_dot)
            
            # Legend Item for Final Scene
            l_dot = Dot(color=r_color)
            l_txt = Text(r_name, font=FONT_FAMILY, font_size=18)
            l_item = VGroup(l_dot, l_txt).arrange(RIGHT)
            final_region_labels.add(l_item)

            # Create/Update Label
            new_label = Text(r_label_txt, font=FONT_FAMILY, font_size=24).move_to(avg_pos + UP * 0.6)
            # Ensure label stays inside screen ? 
            # If too close to top edge, shift down?
            # Basic constraint:
            if new_label.get_top()[1] > 3.5:
                # Shift below dots
                new_label.next_to(active_dots, DOWN)
            
            # Animations
            if i == 0:
                # Initialize
                for idx, dot in enumerate(active_dots):
                    dot.move_to(new_dot_positions[idx])
                state_label = new_label
                self.play(FadeIn(active_dots), Write(state_label))
            else:
                # Transform
                anims = [Transform(state_label, new_label)]
                for idx, dot in enumerate(active_dots):
                    anims.append(dot.animate.move_to(new_dot_positions[idx]))
                self.play(*anims, run_time=1.5)
            
            self.wait(1)
            
        # 4. Final Scene: Show Regional Averages
        # Fade Out Active Model Dots and current Labels
        self.play(FadeOut(active_dots), FadeOut(state_label), FadeOut(legend))
        
        # Arrange Final Legend
        final_region_labels.arrange(DOWN, aligned_edge=LEFT).move_to(legend.get_center())
        
        summary = Text("Average by Dialect (지역별 평균)", font=FONT_FAMILY, font_size=24).to_edge(UP).shift(DOWN * 1.0)
        
        self.play(FadeIn(final_avg_dots), FadeIn(final_region_labels), Write(summary))
        
        self.wait(5)
        
        self.play(FadeOut(Group(*self.mobjects)))

class TruthfulQAScatter(ScatterScene):
    def construct(self):
        # TruthfulQA Range: 0-603
        self.construct_scene("TruthfulQA", 603, "csv_data/TruthfulQA_Accuracy.csv")

class MedNLIScatter(ScatterScene):
    def construct(self):
        # MedNLI Range: 0-1372
        self.construct_scene("MedNLI", 1372, "csv_data/Mednli_Accuracy.csv")
