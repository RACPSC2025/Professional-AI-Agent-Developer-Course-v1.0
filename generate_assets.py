"""
Script para generar assets gráficos en formato SVG para el curso.
Genera banners y diagramas profesionales sin depender de APIs externas.
"""

import os

def create_banner(filename, title, subtitle):
    svg_content = f"""<svg width="1280" height="640" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="100%" height="100%" fill="#0d1117"/>
    <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:#0078D4;stop-opacity:0.2" />
            <stop offset="100%" style="stop-color:#FF9900;stop-opacity:0.1" />
        </linearGradient>
        <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#30363d" stroke-width="1"/>
        </pattern>
    </defs>
    
    <!-- Grid Pattern -->
    <rect width="100%" height="100%" fill="url(#grid)" />
    
    <!-- Gradient Overlay -->
    <rect width="100%" height="100%" fill="url(#grad1)" />
    
    <!-- Decorative Circles -->
    <circle cx="100" cy="100" r="300" fill="#0078D4" fill-opacity="0.05" />
    <circle cx="1180" cy="540" r="200" fill="#FF9900" fill-opacity="0.05" />
    
    <!-- Text -->
    <text x="50%" y="45%" dominant-baseline="middle" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-weight="bold" font-size="60" fill="#ffffff">
        {title}
    </text>
    <text x="50%" y="55%" dominant-baseline="middle" text-anchor="middle" font-family="Segoe UI, Helvetica, Arial, sans-serif" font-size="30" fill="#8b949e">
        {subtitle}
    </text>
    
    <!-- Tech Accents -->
    <rect x="440" y="380" width="400" height="2" fill="#0078D4" />
</svg>"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Generated {filename}")

def create_architecture_diagram(filename):
    svg_content = """<svg width="1000" height="600" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="100%" height="100%" fill="#0d1117"/>
    
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#8b949e" />
        </marker>
    </defs>

    <!-- Styles -->
    <style>
        .box { fill: #161b22; stroke: #30363d; stroke-width: 2; }
        .text { font-family: sans-serif; fill: #c9d1d9; text-anchor: middle; dominant-baseline: middle; }
        .label { font-family: sans-serif; fill: #8b949e; font-size: 14px; text-anchor: middle; }
        .line { stroke: #8b949e; stroke-width: 2; marker-end: url(#arrow); }
        .highlight { stroke: #0078D4; stroke-width: 3; }
    </style>

    <!-- Nodes -->
    <!-- User -->
    <rect x="50" y="250" width="120" height="80" rx="10" class="box" />
    <text x="110" y="290" class="text">User</text>

    <!-- Agent Core -->
    <rect x="300" y="150" width="400" height="300" rx="15" fill="#0d1117" stroke="#0078D4" stroke-width="2" stroke-dasharray="5,5" />
    <text x="500" y="180" class="label" fill="#0078D4">AI AGENT SYSTEM</text>

    <!-- Planner -->
    <rect x="350" y="250" width="120" height="80" rx="5" class="box highlight" />
    <text x="410" y="290" class="text">Planner</text>

    <!-- Memory -->
    <rect x="550" y="250" width="120" height="80" rx="5" class="box" />
    <text x="610" y="290" class="text">Memory</text>

    <!-- LLM -->
    <rect x="450" y="50" width="120" height="60" rx="5" class="box" />
    <text x="510" y="80" class="text">LLM (GPT-4)</text>

    <!-- Tools -->
    <rect x="800" y="250" width="120" height="80" rx="10" class="box" />
    <text x="860" y="290" class="text">Tools</text>

    <!-- Connections -->
    <line x1="170" y1="290" x2="350" y2="290" class="line" /> <!-- User -> Planner -->
    <line x1="470" y1="290" x2="550" y2="290" class="line" /> <!-- Planner -> Memory -->
    <line x1="410" y1="250" x2="450" y2="110" class="line" /> <!-- Planner -> LLM -->
    <line x1="670" y1="290" x2="800" y2="290" class="line" /> <!-- Memory -> Tools -->

</svg>"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Generated {filename}")

def create_module_header(filename, title, color="#0078D4"):
    svg_content = f"""<svg width="1000" height="300" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#0d1117"/>
    <rect width="100%" height="100%" fill="url(#grid)" />
    
    <defs>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#21262d" stroke-width="1"/>
        </pattern>
    </defs>
    
    <rect x="0" y="290" width="1000" height="10" fill="{color}" />
    
    <text x="50" y="150" font-family="Segoe UI, sans-serif" font-weight="bold" font-size="40" fill="#ffffff">
        {title}
    </text>
    <text x="50" y="200" font-family="Segoe UI, sans-serif" font-size="20" fill="#8b949e">
        Professional AI Agent Developer Course
    </text>
</svg>"""
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Generated {filename}")

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    
    # Main Assets
    create_banner("images/course_banner.png", "Professional AI Agent Developer", "From Zero to Enterprise Production")
    create_architecture_diagram("images/architecture_overview.png")
    
    # Module Headers
    create_module_header("images/module0_header.png", "Módulo 0: Fundamentos", "#238636")
    create_module_header("images/module5_header.png", "Módulo 5: Advanced RAG", "#A371F7")
    create_module_header("images/module8_header.png", "Módulo 8: Multi-Agent Systems", "#F778BA")
    create_module_header("images/module12_header.png", "Módulo 12: Capstone Project", "#D29922")
    create_module_header("images/module14_header.png", "Módulo 14: DevOps & Deployment", "#F0883E")
