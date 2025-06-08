import json
import random
import math

def generate_manim_dataset():
    """Generate a comprehensive dataset of scene descriptions and corresponding Manim code"""
    
    dataset = []
    
    # 1. Basic Shape Animations (100 samples)
    shapes = ['Circle', 'Square', 'Triangle', 'Rectangle', 'Polygon']
    colors = ['RED', 'BLUE', 'GREEN', 'YELLOW', 'PURPLE', 'ORANGE', 'PINK', 'CYAN']
    animations = ['FadeIn', 'DrawBorderThenFill', 'Create', 'ShowCreation', 'Write']
    
    for i in range(10):
        shape = random.choice(shapes)
        color = random.choice(colors)
        animation = random.choice(animations)
        
        # Initialize description and code variables
        description = ""
        code = ""
        
        if shape == 'Circle':
            radius = round(random.uniform(0.5, 2.0), 1)
            description = f"A {color.lower()} circle with radius {radius} appears on screen"
            code = f"""from manim import *

class Scene{i+1}(Scene):
    def construct(self):
        circle = Circle(radius={radius}).set_color({color})
        self.play({animation}(circle))
        self.wait(1)"""
        
        elif shape == 'Square':
            side = round(random.uniform(1.0, 3.0), 1)
            description = f"A {color.lower()} square with side length {side} materializes"
            code = f"""from manim import *

class Scene{i+1}(Scene):
    def construct(self):
        square = Square(side_length={side}).set_color({color})
        self.play({animation}(square))
        self.wait(1)"""
        
        elif shape == 'Rectangle':
            width = round(random.uniform(2.0, 4.0), 1)
            height = round(random.uniform(1.0, 2.5), 1)
            description = f"A {color.lower()} rectangle {width}x{height} units draws itself"
            code = f"""from manim import *

class Scene{i+1}(Scene):
    def construct(self):
        rect = Rectangle(width={width}, height={height}).set_color({color})
        self.play({animation}(rect))
        self.wait(1)"""
        
        elif shape == 'Triangle':
            description = f"A {color.lower()} triangle appears with animation"
            code = f"""from manim import *

class Scene{i+1}(Scene):
    def construct(self):
        triangle = Triangle().set_color({color})
        self.play({animation}(triangle))
        self.wait(1)"""
        
        elif shape == 'Polygon':
            sides = random.choice([5, 6, 8])
            description = f"A {color.lower()} {sides}-sided polygon appears on screen"
            code = f"""from manim import *

class Scene{i+1}(Scene):
    def construct(self):
        polygon = RegularPolygon(n={sides}).set_color({color})
        self.play({animation}(polygon))
        self.wait(1)"""
        
        dataset.append({
            "id": i+1,
            "description": description,
            "duration": "1-2 seconds",
            "category": "basic_shapes",
            "manim_code": code
        })
    
    # 2. Text Animations (80 samples)
    text_samples = [
        "Hello World", "Mathematics", "Physics", "Chemistry", "Biology",
        "Welcome", "Python", "Manim", "Animation", "Science",
        "Learning", "Education", "Teaching", "Students", "Knowledge"
    ]
    
    for i in range(8):
        text = random.choice(text_samples)
        color = random.choice(colors)
        font_size = random.choice([24, 36, 48, 60])
        
        description = f"The text '{text}' appears in {color.lower()} color"
        code = f"""from manim import *

class Scene{i+101}(Scene):
    def construct(self):
        text = Text("{text}", font_size={font_size}).set_color({color})
        self.play(Write(text))
        self.wait(1)"""
        
        dataset.append({
            "id": i+101,
            "description": description,
            "duration": "1-2 seconds",
            "category": "text_animation",
            "manim_code": code
        })
    
    # 3. Mathematical Expressions (70 samples)
    expressions = [
        "x^2 + y^2 = r^2", "E = mc^2", "a^2 + b^2 = c^2",
        "\\frac{d}{dx}x^2 = 2x", "\\int x dx = \\frac{x^2}{2}",
        "\\sin^2(x) + \\cos^2(x) = 1", "F = ma", "PV = nRT"
    ]
    
    for i in range(7):
        expr = random.choice(expressions)
        color = random.choice(colors)
        
        description = f"Mathematical equation '{expr}' is written on screen"
        code = f"""from manim import *

class Scene{i+181}(Scene):
    def construct(self):
        equation = MathTex(r"{expr}").set_color({color})
        self.play(Write(equation))
        self.wait(1)"""
        
        dataset.append({
            "id": i+181,
            "description": description,
            "duration": "1-2 seconds",
            "category": "math_expressions",
            "manim_code": code
        })
    
    # 4. Movement Animations (60 samples)
    directions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
    
    for i in range(600):
        shape = random.choice(['Circle', 'Square'])
        color = random.choice(colors)
        direction = random.choice(directions)
        distance = round(random.uniform(1.0, 3.0), 1)
        
        description = f"A {color.lower()} {shape.lower()} moves {direction.lower()} by {distance} units"
        
        if shape == 'Circle':
            code = f"""from manim import *

class Scene{i+251}(Scene):
    def construct(self):
        circle = Circle().set_color({color})
        self.play(Create(circle))
        self.play(circle.animate.shift({direction} * {distance}))
        self.wait(0.5)"""
        else:
            code = f"""from manim import *

class Scene{i+251}(Scene):
    def construct(self):
        square = Square().set_color({color})
        self.play(Create(square))
        self.play(square.animate.shift({direction} * {distance}))
        self.wait(0.5)"""
        
        dataset.append({
            "id": i+251,
            "description": description,
            "duration": "1-2 seconds",
            "category": "movement",
            "manim_code": code
        })
    
    # 5. Rotation Animations (50 samples)
    for i in range(5):
        shape = random.choice(['Square', 'Rectangle', 'Triangle'])
        color = random.choice(colors)
        angle = random.choice([45, 90, 180, 270, 360])
        
        description = f"A {color.lower()} {shape.lower()} rotates {angle} degrees"
        
        if shape == 'Square':
            code = f"""from manim import *

class Scene{i+311}(Scene):
    def construct(self):
        square = Square().set_color({color})
        self.play(Create(square))
        self.play(Rotate(square, {angle}*DEGREES))
        self.wait(0.5)"""
        elif shape == 'Rectangle':
            code = f"""from manim import *

class Scene{i+311}(Scene):
    def construct(self):
        rect = Rectangle().set_color({color})
        self.play(Create(rect))
        self.play(Rotate(rect, {angle}*DEGREES))
        self.wait(0.5)"""
        else:  # Triangle
            code = f"""from manim import *

class Scene{i+311}(Scene):
    def construct(self):
        triangle = Triangle().set_color({color})
        self.play(Create(triangle))
        self.play(Rotate(triangle, {angle}*DEGREES))
        self.wait(0.5)"""
        
        dataset.append({
            "id": i+311,
            "description": description,
            "duration": "1-2 seconds",
            "category": "rotation",
            "manim_code": code
        })
    
    # 6. Scaling Animations (40 samples)
    for i in range(4):
        shape = random.choice(['Circle', 'Square'])
        color = random.choice(colors)
        scale_factor = round(random.uniform(0.5, 2.5), 1)
        
        description = f"A {color.lower()} {shape.lower()} scales by factor {scale_factor}"
        
        if shape == 'Circle':
            code = f"""from manim import *

class Scene{i+361}(Scene):
    def construct(self):
        circle = Circle().set_color({color})
        self.play(Create(circle))
        self.play(circle.animate.scale({scale_factor}))
        self.wait(0.5)"""
        else:
            code = f"""from manim import *

class Scene{i+361}(Scene):
    def construct(self):
        square = Square().set_color({color})
        self.play(Create(square))
        self.play(square.animate.scale({scale_factor}))
        self.wait(0.5)"""
        
        dataset.append({
            "id": i+361,
            "description": description,
            "duration": "1-2 seconds",
            "category": "scaling",
            "manim_code": code
        })
    
    # 7. Graph and Function Plotting (50 samples)
    functions = [
        ("x^2", "x**2", "parabola"),
        ("sin(x)", "np.sin(x)", "sine wave"),
        ("cos(x)", "np.cos(x)", "cosine wave"),
        ("x^3", "x**3", "cubic function"),
        ("e^x", "np.exp(x)", "exponential function")
    ]
    
    for i in range(5):
        func_tex, func_code, func_name = random.choice(functions)
        color = random.choice(colors)
        
        description = f"A {color.lower()} graph of {func_name} {func_tex} appears"
        code = f"""from manim import *
import numpy as np

class Scene{i+401}(Scene):
    def construct(self):
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3])
        graph = axes.plot(lambda x: {func_code}, color={color})
        self.play(Create(axes))
        self.play(Create(graph))
        self.wait(1)"""
        
        dataset.append({
            "id": i+401,
            "description": description,
            "duration": "2 seconds",
            "category": "graphs",
            "manim_code": code
        })
    
    # 8. Transformation Animations (50 samples)
    transformations = [
        ("Circle", "Square", "circle transforms into square"),
        ("Square", "Triangle", "square morphs into triangle"),
        ("Triangle", "Circle", "triangle becomes circle"),
        ("Rectangle", "Circle", "rectangle transforms to circle")
    ]
    
    for i in range(5):
        shape1, shape2, description_text = random.choice(transformations)
        color = random.choice(colors)
        
        description = f"A {color.lower()} {description_text}"
        code = f"""from manim import *

class Scene{i+451}(Scene):
    def construct(self):
        shape1 = {shape1}().set_color({color})
        shape2 = {shape2}().set_color({color})
        self.play(Create(shape1))
        self.play(Transform(shape1, shape2))
        self.wait(1)"""
        
        dataset.append({
            "id": i+451,
            "description": description,
            "duration": "2 seconds",
            "category": "transformations",
            "manim_code": code
        })
    
    return dataset

# Generate the dataset
dataset = generate_manim_dataset()

# Save as JSON file
with open('manim_scene_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Dataset generated with {len(dataset)} samples")
print("Categories distribution:")
categories = {}
for item in dataset:
    cat = item['category']
    categories[cat] = categories.get(cat, 0) + 1

for cat, count in categories.items():
    print(f"  {cat}: {count} samples")

print("\nDataset saved as 'manim_scene_dataset.json'")
print("Sample structure:")
print(json.dumps(dataset[0], indent=2))