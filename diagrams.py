import pandas as pd

# %%

#Diagram list

# 1. Mode to mode sankey diagram (ie input mode, representation mode (including joint embedding space?), output mode) (Eg: https://www.epirhandbook.com/en/new_pages/diagrams.html#alluvialsankey-diagrams)
# 2. audio sample, annotated to varying degrees.
# 3. dataset metadata type and how much of that gets used in training 
# 

# %%
# Sankey diagram of input mode, representation mode, output mode for different models.
import plotly.graph_objects as go
import pandas as pd
import re

# Load data
df = pd.read_csv("responsible_ai_models.csv")

# Filter out models that don't output audio
df = df[df['Output Type'].str.contains('Audio', na=False, case=False)]

# Filter out spectrogram models
rep_col = 'Representation Class' if 'Representation Class' in df.columns else 'Representation'
if rep_col in df.columns:
    df = df[~df[rep_col].astype(str).str.contains('spectrogram', case=False, na=False)]

# Remove MIDI from the Output Type column to drop the MIDI output node
df['Output Type'] = df['Output Type'].str.replace(r'(?i)midi', '', regex=True)
df['Output Type'] = df['Output Type'].str.replace(r'/', ',', regex=True)
df['Output Type'] = df['Output Type'].str.replace(r',\s*,', ',', regex=True).str.strip(' ,')

# Combine Class and Subclass into a single Representation column
# (Safely handling any numeric/float NaN values to avoid iter/float errors!)

# Fix missing column names gracefully in case the CSV uses original column headers
if 'Intermediate Representation' in df.columns and 'Representation Class' not in df.columns:
    df['Representation Class'] = df['Intermediate Representation']
if 'Representation Subclass' not in df.columns:
    df['Representation Subclass'] = 'N/S'
if 'Additional Conditioning & Controls' in df.columns and 'Conditioning & Controls' not in df.columns:
    df['Conditioning & Controls'] = df['Additional Conditioning & Controls']

def format_rep(row):
    # Convert anything to string safely or default to 'N/S' if empty/NaN
    r_class = str(row['Representation Class']) if pd.notna(row['Representation Class']) else 'N/S'
    
    return r_class

df['Representation'] = df.apply(format_rep, axis=1)

# Define the exact flow of columns you want in your Sankey diagram.
cols = [
    'Input Type', 
    'Representation', 
    'Conditioning & Controls', 
    'Output Type'
]

df_sankey = df[cols].copy()

# Explode each column so we get individual paths rather than merged strings for sankey linking
def explode_column(tdf, col):
    tdf[col] = tdf[col].astype(str)
    s = tdf[col].str.split(',').explode()
    s = s.str.strip()
    return tdf.drop(col, axis=1).join(s)

for col in cols:
    df_sankey = explode_column(df_sankey, col)

# Drop any rows that became 'nan' or empty
df_sankey = df_sankey.replace({'nan': 'N/S', '': 'N/S'})

# Group specific conditioning and controls
group_mapping = {
    'Key': 'Musical Structure',
    'Structure': 'Musical Structure',
    'Tempo': 'Musical Structure',
    'Melody Chroma': 'Musical Structure',
    'Energy': 'Emotion',
    'Mood': 'Emotion',
    'Emotion Label': 'Emotion'
}
df_sankey = df_sankey.replace(group_mapping)

# Append column names or suffixes to node values to avoid cycles / loops between layers
for col in cols:
    df_sankey[col] = df_sankey[col] + f" ({col})"

# Build connections dynamically to allow skipping empty 'Conditioning & Controls'
link_records = []
for _, row in df_sankey.iterrows():
    # 1. Input Type -> Representation
    link_records.append({'source': row['Input Type'], 'target': row['Representation']})
    
    # 2. Representation -> Output (if no conditioning) OR -> Conditioning -> Output
    if str(row['Conditioning & Controls']).startswith('N/S'):
        link_records.append({'source': row['Representation'], 'target': row['Output Type']})
    else:
        link_records.append({'source': row['Representation'], 'target': row['Conditioning & Controls']})
        link_records.append({'source': row['Conditioning & Controls'], 'target': row['Output Type']})

links_df = pd.DataFrame(link_records)
layer_links = links_df.groupby(['source', 'target']).size().reset_index(name='value')

# Extract all unique nodes and map them to indices from the actual generated links
nodes = pd.concat([links_df['source'], links_df['target']]).unique()
node_dict = {node: i for i, node in enumerate(nodes)}

# Map links to indices
links = layer_links.copy()
links['source'] = links['source'].map(node_dict)
links['target'] = links['target'].map(node_dict)

# Create clean labels (remove the layer suffixes for display)
labels = [str(node).split(" (")[0] for node in nodes]
labels = ["" if l == "N/S" else l for l in labels]

sankey_colors = {
    'genre': '#EF553B',     
    'timbre': '#00CC96',    
    'musical structure': '#AB63FA',  
    'text': '#1f77b4'
}

bright_colors = [
    '#FFA15A', '#19D3F3', 
    '#FF6692', '#B6E880', '#FF97FF', '#1f77b4', 
    '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'
]

node_colors = []

for i, label in enumerate(labels):
    clean_label = label.strip().lower() if label else ""

    if clean_label in ["", "none", "unknown"]:
        node_colors.append('#D3D3D3')  # Light gray
    elif clean_label in sankey_colors:   
        print('colour is:', clean_label)
        node_colors.append(sankey_colors[clean_label])
    else:
        node_colors.append(bright_colors[i % len(bright_colors)])

# Create matching transparent colors for links based on their source node
link_colors = []
for src in links['source']:
    hex_c = node_colors[src].lstrip('#')
    link_colors.append(f"rgba({int(hex_c[0:2], 16)}, {int(hex_c[2:4], 16)}, {int(hex_c[4:6], 16)}, 0.4)")

# Create and show Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 35,
      thickness = 25,
      line = dict(color = "white", width = 0.5),
      label = labels,
      color = node_colors
    ),
    link = dict(
      source = links['source'],
      target = links['target'],
      value = links['value'],
      color = link_colors
    ))])

title_path = " -> ".join([c.replace('Type', '').strip() for c in cols])
fig.update_layout(
    height=1200,
    title=dict(
        text=f"<b>Representation and Control Strategies for Generative Audio Models</b>",
        x=0.5,
        y=0.98,
        font=dict(size=36, color="#21386a", family="sans-serif")
    ),
    font=dict(size=20, family="sans-serif", color="black"),
    margin=dict(t=220, l=50, r=50, b=50),
    plot_bgcolor='white',
    paper_bgcolor='white',
    annotations=[
        dict(x=0.0, y=1.1, text='<b>Input Type</b>', showarrow=False, xref='paper', yref='paper', xanchor='left', font=dict(size=24, color="#21386a")),
        dict(x=0.33, y=1.1, text='<b>Internal Representation Type</b>', showarrow=False, xref='paper', yref='paper', xanchor='center', font=dict(size=24, color="#21386a")),
        dict(x=0.67, y=1.1, text='<b>Additional Conditioning, Controls<br>or Latent Structure</b>', showarrow=False, xref='paper', yref='paper', xanchor='center', font=dict(size=24, color="#21386a")),
        dict(x=1.0, y=1.1, text='<b>Output Type</b>', showarrow=False, xref='paper', yref='paper', xanchor='right', font=dict(size=24, color="#21386a"))
    ]
)
fig.show()

