import json
from pathlib import Path

# Load AST
ast_data = json.loads(Path("graphify-out/.graphify_ast.json").read_text())

# Load semantic
semantic_data = json.loads(Path("graphify-out/chunk_01.json").read_text())

# Merge: AST nodes first, semantic nodes deduplicated
seen = {n['id'] for n in ast_data['nodes']}
merged_nodes = list(ast_data['nodes'])
for n in semantic_data['nodes']:
    if n['id'] not in seen:
        merged_nodes.append(n)
        seen.add(n['id'])

# Combine edges and hyperedges
merged_edges = ast_data['edges'] + semantic_data['edges']
merged_hyperedges = semantic_data.get('hyperedges', [])

# Merge token counts
merged = {
    'nodes': merged_nodes,
    'edges': merged_edges,
    'hyperedges': merged_hyperedges,
    'input_tokens': semantic_data.get('input_tokens', 0),
    'output_tokens': semantic_data.get('output_tokens', 0),
}

Path('graphify-out/.graphify_extract.json').write_text(json.dumps(merged, indent=2))
print(f'Merged: {len(merged_nodes)} nodes, {len(merged_edges)} edges')
