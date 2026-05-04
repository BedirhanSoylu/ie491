import json
from pathlib import Path
from graphify.build import build_from_json
from graphify.cluster import score_all
from graphify.analyze import suggest_questions
from graphify.report import generate
from graphify.export import to_html

# Load data
extraction = json.loads(Path('graphify-out/.graphify_extract.json').read_text())
detection = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
analysis = json.loads(Path('graphify-out/.graphify_analysis.json').read_text())

# Rebuild graph
G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis['cohesion'].items()}
tokens = {'input': extraction.get('input_tokens', 0), 'output': extraction.get('output_tokens', 0)}

# Label communities based on analysis
labels = {
    0: "RANSAC Edge Detection",
    1: "Force Error Calculation",
    2: "Experimental Data Analysis",
    3: "Tool Wear Measurement",
    4: "B-Spline Optimization",
    5: "calculateForceError Module",
    6: "computeResiduals Module",
    7: "ForceAnalysis v11 Module",
    8: "ForceAnalysis v13 Module",
    9: "Spline Optimizer Module",
    10: "Worn Area Module",
}

# Regenerate questions with real labels
questions = suggest_questions(G, communities, labels)

# Regenerate report with labels
report = generate(G, communities, cohesion, labels, analysis['gods'], analysis['surprises'], detection, tokens, '.', suggested_questions=questions)
Path('graphify-out/GRAPH_REPORT.md').write_text(report, encoding='utf-8')

# Save labels
Path('graphify-out/.graphify_labels.json').write_text(json.dumps({str(k): v for k, v in labels.items()}))

print('Communities labeled and report updated')

# Generate HTML visualization
if G.number_of_nodes() <= 5000:
    to_html(G, communities, 'graphify-out/graph.html', community_labels=labels or None)
    print('graph.html written - open in any browser')
else:
    print(f'Graph too large ({G.number_of_nodes()} nodes) for HTML visualization')
