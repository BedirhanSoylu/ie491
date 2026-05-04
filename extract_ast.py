import sys, json
from graphify.extract import collect_files, extract
from pathlib import Path

code_files = [
    "calculateForceError_Fixed.m",
    "computeResiduals_Spline.m",
    "ForceDataAnalysisv11tool15.m",
    "ForceDataAnalysisv13tool15.m",
    "run_Spline_Optimizer.m",
    "wornArea.m"
]
code_files = [Path(f) for f in code_files if Path(f).exists()]

if code_files:
    result = extract(code_files)
    Path("graphify-out/.graphify_ast.json").write_text(json.dumps(result, indent=2))
    print(f"AST: {len(result['nodes'])} nodes, {len(result['edges'])} edges")
else:
    Path("graphify-out/.graphify_ast.json").write_text(json.dumps({"nodes":[],"edges":[],"input_tokens":0,"output_tokens":0}))
    print("No code files found")
