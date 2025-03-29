import streamlit as st
import requests
import json
import subprocess
import sys
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go
import pandas as pd
import logging
import random
from io import BytesIO
from datetime import datetime
from math import cos, sin, pi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="🧬 Advanced Protein Explorer Pro",
    page_icon="🧬",
    layout="wide"
)

# Initialize session state for persistent settings
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'last_search' not in st.session_state:
    st.session_state.last_search = ""
if 'protein_info' not in st.session_state:
    st.session_state.protein_info = None

# List of random proteins for daily cycling
PROTEIN_LIST = ["Hemoglobin", "Insulin", "Myoglobin", "Collagen", "Ferritin", "Cytochrome C", "Albumin", "Keratin"]

# Amino acid weights dictionary
AA_WEIGHTS = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'E': 147.1, 'Q': 146.2, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}

# Amino acid properties
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'group': 'Aliphatic'},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'group': 'Positively Charged'},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'group': 'Polar Uncharged'},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'group': 'Negatively Charged'},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'group': 'Special Cases'},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'group': 'Negatively Charged'},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'group': 'Polar Uncharged'},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'group': 'Special Cases'},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'group': 'Positively Charged'},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'group': 'Aliphatic'},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'group': 'Aliphatic'},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'group': 'Positively Charged'},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'group': 'Aliphatic'},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'group': 'Aromatic'},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'group': 'Special Cases'},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'group': 'Polar Uncharged'},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'group': 'Polar Uncharged'},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'group': 'Aromatic'},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'group': 'Aromatic'},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'group': 'Aliphatic'}
}

def get_daily_protein():
    """Selects a daily protein based on the current date"""
    random.seed(datetime.now().date().toordinal())
    return random.choice(PROTEIN_LIST)

# Use checkbox instead of toggle for dark mode
st.session_state.dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)

# Apply comprehensive dark mode styling if enabled
if st.session_state.dark_mode:
    # More comprehensive dark mode styling
    st.markdown("""
    <style>
    /* Main background and text */
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #252526;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #4a69bd !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4a69bd;
        color: white;
        border: none;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: white;
        border: 1px solid #4a69bd;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: #2D2D2D;
        color: white;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div > div {
        background-color: #2D2D2D;
        color: white;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #2D2D2D;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #2D2D2D;
        color: white;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #252526;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4a69bd;
    }
    
    /* Success/Info/Warning/Error messages */
    .element-container .stAlert {
        background-color: #2D2D2D;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Set darker background for plots
    plot_bg_color = "#1E1E1E"
    plot_font_color = "#FFFFFF"
    plot_grid_color = "#333333"
else:
    # Light mode settings for plots
    plot_bg_color = "#FFFFFF"
    plot_font_color = "#000000"
    plot_grid_color = "#E5E5E5"

@st.cache_data
def get_protein_info(protein_name):
    """Fetch detailed protein information from UniProt API and cache results"""
    try:
        search_url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=json"
        response = requests.get(search_url)
        
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}: {response.text}")
            return None
            
        data = response.json()
        
        if 'results' in data and data['results']:
            result = data['results'][0]
            
            # Safely extract nested values with default fallbacks
            return {
                'name': result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown'),
                'function': next((c.get('value') for c in result.get('comments', []) if c.get('type') == 'FUNCTION'), 'Function unknown'),
                'gene': result.get('genes', [{}])[0].get('geneName', {}).get('value', 'Unknown') if result.get('genes') else 'Unknown',
                'organism': result.get('organism', {}).get('scientificName', 'Unknown'),
                'length': result.get('sequence', {}).get('length', 'Unknown'),
                'mass': result.get('sequence', {}).get('mass', 'Unknown'),
                'subcellular_locations': [loc.get('value') for loc in result.get('subcellularLocations', [])],
                'sequence': result.get('sequence', {}).get('value', 'Sequence not available'),
                'pdb_id': next((xref.get('id') for xref in result.get('uniProtKBCrossReferences', []) 
                               if xref.get('database') == 'PDB'), None)
            }
        logger.warning(f"No results found for protein: {protein_name}")
        return None
    except Exception as e:
        logger.error(f"Error fetching protein information: {str(e)}", exc_info=True)
        return None

def plot_amino_acid_composition(sequence):
    """Plot amino acid composition of the protein"""
    if not sequence:
        return None
    aa_count = {aa: sequence.count(aa) for aa in set(sequence)}
    aa_sorted = sorted(aa_count.items(), key=lambda x: x[1], reverse=True)
    
    fig = go.Figure(data=[go.Bar(x=[aa[0] for aa in aa_sorted], 
                                 y=[aa[1] for aa in aa_sorted],
                                 text=[aa[1] for aa in aa_sorted], 
                                 textposition='auto')])
    fig.update_layout(
        title="Amino Acid Composition", 
        xaxis_title="Amino Acid", 
        yaxis_title="Count",
        paper_bgcolor=plot_bg_color,
        plot_bgcolor=plot_bg_color,
        font=dict(color=plot_font_color)
    )
    fig.update_xaxes(gridcolor=plot_grid_color)
    fig.update_yaxes(gridcolor=plot_grid_color)
    return fig

def download_fasta(protein_name, sequence):
    """Download protein sequence in FASTA format"""
    fasta_format = f">{protein_name}\n{sequence}"
    return BytesIO(fasta_format.encode())

def calculate_protein_properties(sequence):
    """Calculate basic physicochemical properties of the protein sequence"""
    if not sequence:
        return None
        
    # Amino acid properties
    hydrophobic = set('AILMFWYV')
    polar = set('QNSTGH')
    charged = set('KRHDE')
    
    seq_len = len(sequence)
    if seq_len == 0:
        return None
        
    properties = {
        'length': seq_len,
        'hydrophobic_ratio': len([aa for aa in sequence if aa in hydrophobic]) / seq_len,
        'polar_ratio': len([aa for aa in sequence if aa in polar]) / seq_len,
        'charged_ratio': len([aa for aa in sequence if aa in charged]) / seq_len,
        'molecular_weight': sum(AA_WEIGHTS.get(aa, 0) for aa in sequence)
    }
    return properties

def predict_secondary_structure(sequence):
    """Simple secondary structure prediction based on amino acid propensities"""
    if not sequence:
        return ""
        
    # Propensity values for different secondary structures
    helix_prone = set('AEILMQ')
    sheet_prone = set('VIVYW')
    turn_prone = set('NPGS')
    
    structure = []
    for i in range(len(sequence)):
        if sequence[i] in helix_prone:
            structure.append('H')  # Helix
        elif sequence[i] in sheet_prone:
            structure.append('E')  # Sheet
        elif sequence[i] in turn_prone:
            structure.append('T')  # Turn
        else:
            structure.append('C')  # Coil
    
    return ''.join(structure)

def analyze_conservation(sequence):
    """Analyze sequence conservation patterns"""
    if not sequence:
        return None
        
    # Dictionary of amino acid groups based on properties
    aa_groups = {
        'hydrophobic': set('AILMFWYV'),
        'polar': set('QNSTGH'),
        'positive': set('KRH'),
        'negative': set('DE'),
        'special': set('CP')
    }
    
    # Count occurrences of each group
    group_counts = {group: sum(1 for aa in sequence if aa in aas) 
                   for group, aas in aa_groups.items()}
    
    return group_counts

def calculate_structure_properties(pdb_id):
    """Calculate structural properties from PDB data"""
    if not pdb_id:
        return None
        
    try:
        # Simulate structure calculations (in real app, would fetch from PDB API)
        properties = {
            'resolution': round(random.uniform(1.5, 3.0), 2),
            'r_value': round(random.uniform(0.15, 0.25), 3),
            'chains': random.randint(1, 4),
            'residues': random.randint(100, 500),
            'secondary_structure': {
                'alpha_helix': round(random.uniform(30, 60), 1),
                'beta_sheet': round(random.uniform(10, 40), 1),
                'loops': round(random.uniform(10, 30), 1)
            }
        }
        return properties
    except Exception as e:
        logger.error(f"Error calculating structure properties: {str(e)}", exc_info=True)
        return None

def predict_binding_sites(sequence):
    """Predict potential binding sites based on sequence patterns"""
    if not sequence:
        return []
        
    # Simple binding site prediction based on amino acid properties
    binding_sites = []
    window_size = 5
    
    # Define binding-prone residues
    binding_prone = set('RHKDESTNQ')  # Polar and charged residues
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        # Count binding-prone residues in window
        binding_score = sum(1 for aa in window if aa in binding_prone)
        if binding_score >= 3:  # If more than half residues are binding-prone
            binding_sites.append({
                'position': i,
                'sequence': window,
                'score': binding_score / window_size
            })
    
    return binding_sites

def analyze_domains(sequence):
    """Predict protein domains based on sequence patterns"""
    if not sequence:
        return []
        
    domains = []
    min_domain_length = 50
    current_length = 0
    current_start = 0
    
    # Simple domain prediction based on hydrophobicity patterns
    hydrophobic = set('AILMFWYV')
    
    for i, aa in enumerate(sequence):
        if aa in hydrophobic:
            current_length += 1
        else:
            if current_length >= min_domain_length:
                domains.append({
                    'start': current_start,
                    'end': i,
                    'length': current_length,
                    'sequence': sequence[current_start:i]
                })
            current_length = 0
            current_start = i + 1
    
    return domains

def plot_aa_property_distribution(sequence):
    """Plot amino acid property distribution for the protein"""
    if not sequence:
        return None
    
    # Group amino acids by their properties
    aa_groups = {
        'Aliphatic': 0,
        'Aromatic': 0,
        'Positively Charged': 0,
        'Negatively Charged': 0,
        'Polar Uncharged': 0,
        'Special Cases': 0
    }
    
    for aa in sequence:
        if aa in AA_PROPERTIES:
            group = AA_PROPERTIES[aa]['group']
            aa_groups[group] += 1
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(aa_groups.keys()),
        values=list(aa_groups.values()),
        hole=.3
    )])
    fig.update_layout(
        title="Amino Acid Group Distribution",
        paper_bgcolor=plot_bg_color,
        plot_bgcolor=plot_bg_color,
        font=dict(color=plot_font_color)
    )
    return fig

def calculate_hydropathy_plot(sequence, window_size=7):
    """Calculate hydropathy values for a sliding window across the sequence"""
    if not sequence or window_size > len(sequence):
        return None
    
    hydropathy_values = []
    positions = []
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i+window_size]
        avg_hydropathy = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in window) / window_size
        hydropathy_values.append(avg_hydropathy)
        positions.append(i + (window_size // 2))
    
    return positions, hydropathy_values

def render_composition_tab(tabs, protein_info):
    """Render the composition tab content"""
    if not protein_info:
        return
        
    with tabs[1]:  # Composition
        st.markdown("### Amino Acid Composition Analysis")
        
        sequence = protein_info.get('sequence', '')
        if not sequence:
            st.warning("Protein sequence not available for analysis.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot amino acid distribution by count
            composition_fig = plot_amino_acid_composition(sequence)
            if composition_fig:
                st.plotly_chart(composition_fig)
                
            # Calculate amino acid percentages
            aa_count = {aa: sequence.count(aa) for aa in set(sequence)}
            total_count = len(sequence)
            aa_percentage = {aa: (count / total_count) * 100 for aa, count in aa_count.items()}
            
            # Display percentages
            st.markdown("#### Amino Acid Percentages")
            percentage_data = pd.DataFrame({
                'Amino Acid': list(aa_percentage.keys()),
                'Percentage (%)': [round(p, 2) for p in aa_percentage.values()]
            }).sort_values(by='Percentage (%)', ascending=False)
            
            st.dataframe(percentage_data)
            
        with col2:
            # Amino acid group distribution
            st.markdown("#### Amino Acid Group Distribution")
            group_fig = plot_aa_property_distribution(sequence)
            if group_fig:
                st.plotly_chart(group_fig)
            
            # Calculate protein properties
            properties = calculate_protein_properties(sequence)
            if properties:
                st.markdown("#### Physicochemical Properties")
                st.write(f"Total length: {properties['length']} amino acids")
                st.write(f"Molecular weight: {properties['molecular_weight']:.1f} Da")
                
                # Create gauge charts for property ratios
                hydrophobic_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = properties['hydrophobic_ratio'] * 100,
                    title = {'text': "Hydrophobic Content (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "darkgray"}
                        ]
                    }
                ))
                hydrophobic_fig.update_layout(height=250, paper_bgcolor=plot_bg_color, font=dict(color=plot_font_color))
                
                charged_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = properties['charged_ratio'] * 100,
                    title = {'text': "Charged Content (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "gray"},
                            {'range': [70, 100], 'color': "darkgray"}
                        ]
                    }
                ))
                charged_fig.update_layout(height=250, paper_bgcolor=plot_bg_color, font=dict(color=plot_font_color))
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.plotly_chart(hydrophobic_fig)
                with col2b:
                    st.plotly_chart(charged_fig)
        
        # Hydropathy plot
        st.markdown("### Hydropathy Analysis")
        col3, col4 = st.columns([3, 1])
        
        with col4:
            window_size = st.slider("Window Size", min_value=5, max_value=21, value=7, step=2)
        
        with col3:
            positions, hydropathy_values = calculate_hydropathy_plot(sequence, window_size)
            if positions and hydropathy_values:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=positions, 
                    y=hydropathy_values,
                    mode='lines',
                    name='Hydropathy'
                ))
                
                # Add zero line
                fig.add_shape(
                    type="line",
                    x0=min(positions),
                    y0=0,
                    x1=max(positions),
                    y1=0,
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash",
                    )
                )
                
                fig.update_layout(
                    title=f"Kyte-Doolittle Hydropathy Plot (Window Size: {window_size})",
                    xaxis_title="Amino Acid Position",
                    yaxis_title="Hydropathy Score",
                    paper_bgcolor=plot_bg_color,
                    plot_bgcolor=plot_bg_color,
                    font=dict(color=plot_font_color)
                )
                fig.update_xaxes(gridcolor=plot_grid_color)
                fig.update_yaxes(gridcolor=plot_grid_color)
                
                st.plotly_chart(fig)
                
                st.markdown("""
                **Interpretation**:
                - Positive values (above the red line) indicate hydrophobic regions
                - Negative values (below the red line) indicate hydrophilic regions
                - Peaks above 1.8 often indicate potential transmembrane regions
                - Valleys below -1.8 often indicate surface-exposed regions
                """)
            else:
                st.warning("Unable to generate hydropathy plot.")
        
        # Export options
        st.markdown("### Export Composition Analysis")
        if st.button("Export Composition Data"):
            # Create CSV with composition data
            aa_data = {aa: sequence.count(aa) for aa in set(sequence)}
            aa_df = pd.DataFrame({
                'Amino Acid': list(aa_data.keys()),
                'Count': list(aa_data.values()),
                'Percentage': [round((count / len(sequence)) * 100, 2) for count in aa_data.values()]
            }).sort_values(by='Count', ascending=False)
            
            csv = aa_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Composition CSV",
                csv,
                file_name=f"{protein_info.get('name', 'Unknown')}_composition.csv",
                mime="text/csv"
            )

def render_structure_tab(tabs, protein_info):
    """Render the enhanced structure tab content"""
    if not protein_info:
        return
        
    with tabs[2]:  # Structure
        st.markdown("### Protein Structure")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            pdb_id = protein_info.get('pdb_id')
            if pdb_id:
                st.image(f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_assembly-1.jpeg",
                         caption=f"Structure of {protein_info['name']} (PDB ID: {pdb_id})")
                
                # Add 3D structure viewer options
                st.markdown("#### 3D Structure Visualization")
                view_options = st.multiselect("Display options:", 
                    ["Cartoon", "Surface", "Ball and Stick"],
                    default=["Cartoon"])
                st.markdown(f"[🔬 View Interactive 3D Structure on PDB](https://www.rcsb.org/3d-view/{pdb_id})")
                
                # Calculate and display structure properties
                if st.button("Calculate Structure Properties"):
                    with st.spinner("Calculating structure properties..."):
                        properties = calculate_structure_properties(pdb_id)
                        if properties:
                            st.markdown("#### Structure Properties")
                            st.write(f"Resolution: {properties['resolution']} Å")
                            st.write(f"R-value: {properties['r_value']}")
                            st.write(f"Number of chains: {properties['chains']}")
                            st.write(f"Total residues: {properties['residues']}")
                            
                            # Create secondary structure composition pie chart
                            fig = go.Figure(data=[go.Pie(
                                labels=list(properties['secondary_structure'].keys()),
                                values=list(properties['secondary_structure'].values()),
                                hole=.3
                            )])
                            fig.update_layout(
                                title="Secondary Structure Composition",
                                paper_bgcolor=plot_bg_color,
                                plot_bgcolor=plot_bg_color,
                                font=dict(color=plot_font_color)
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("Unable to calculate structure properties.")
            else:
                st.info("No structure available for this protein in the PDB.")
                st.markdown("#### Structure Prediction")
                st.write("Consider using structure prediction tools:")
                st.markdown("- [AlphaFold DB](https://alphafold.ebi.ac.uk/)")
                st.markdown("- [Swiss-Model](https://swissmodel.expasy.org/)")
        
        with col2:
            st.markdown("### Structure Analysis Tools")
            
            sequence = protein_info.get('sequence', '')
            if not sequence:
                st.warning("Protein sequence not available for analysis.")
                return
                
            # Binding site prediction
            if st.button("Identify Binding Sites"):
                with st.spinner("Identifying binding sites..."):
                    binding_sites = predict_binding_sites(sequence)
                    if binding_sites:
                        st.markdown("#### Predicted Binding Sites")
                        for site in binding_sites:
                            st.write(f"Position {site['position']}-{site['position']+5}:")
                            st.write(f"Sequence: {site['sequence']}")
                            st.write(f"Score: {site['score']:.2f}")
                            st.markdown("---")
                    else:
                        st.info("No significant binding sites predicted.")
            
            # Domain analysis
            if st.button("Analyze Structural Domains"):
                with st.spinner("Analyzing domains..."):
                    domains = analyze_domains(sequence)
                    if domains:
                        st.markdown("#### Predicted Domains")
                        for i, domain in enumerate(domains, 1):
                            st.write(f"Domain {i}:")
                            st.write(f"Region: {domain['start']}-{domain['end']}")
                            st.write(f"Length: {domain['length']} residues")
                            st.markdown("---")
                    else:
                        st.info("No significant domains predicted.")
                        
            # Surface analysis option
            if st.button("Calculate Surface Properties"):
                with st.spinner("Calculating surface properties..."):
                    st.markdown("#### Surface Properties")
                    # Calculate exposed surface area
                    total_residues = len(sequence)
                    exposed = sum(1 for aa in sequence if aa in 'RKDENQ')
                    surface_ratio = exposed / total_residues
                    
                    # Create gauge chart for surface exposure
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = surface_ratio * 100,
                        title = {'text': "Surface Exposure"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "royalblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "gray"},
                                {'range': [70, 100], 'color': "darkgray"}
                            ]
                        }
                    ))
                    fig.update_layout(
                        paper_bgcolor=plot_bg_color,
                        font=dict(color=plot_font_color)
                    )
                    st.plotly_chart(fig)

def predict_protein_interactions(protein_info):
    """Simulate protein-protein interaction prediction"""
    if not protein_info:
        return []
        
    # In a real app, this would call an external API or use a model
    # For demonstration, we'll create simulated interaction data
    common_interactors = {
        "Hemoglobin": ["Haptoglobin", "Transferrin", "Albumin"],
        "Insulin": ["Insulin Receptor", "IGF1", "Glucagon"],
        "Myoglobin": ["Cytochrome C", "Cytochrome Oxidase", "Hemoglobin"],
        "Collagen": ["Integrin", "Fibronectin", "Elastin"],
        "Ferritin": ["Transferrin Receptor", "DMT1", "Ferroportin"],
        "Cytochrome C": ["Cytochrome Oxidase", "Apaf-1", "Bcl-xL"],
        "Albumin": ["Fatty Acids", "Bilirubin", "Calcium"],
        "Keratin": ["Filaggrin", "Involucrin", "Loricrin"]
    }
    
    protein_name = protein_info.get('name', '')
    
    # Find similar proteins if exact match not available
    for key in common_interactors.keys():
        if key.lower() in protein_name.lower():
            protein_name = key
            break
    
    interactions = []
    if protein_name in common_interactors:
        base_interactors = common_interactors[protein_name]
        
        # Add some randomness to confidence scores
        for interactor in base_interactors:
            interactions.append({
                'protein': interactor,
                'confidence': round(random.uniform(0.6, 0.95), 2),
                'evidence': random.choice(['Experimental', 'Text-mining', 'Database', 'Co-expression'])
            })
        
        # Add a few more random interactions
        all_proteins = [p for p in common_interactors.keys() if p != protein_name]
        random_interactors = random.sample(all_proteins, min(3, len(all_proteins)))
        
        for interactor in random_interactors:
            interactions.append({
                'protein': interactor,
                'confidence': round(random.uniform(0.3, 0.7), 2),
                'evidence': random.choice(['Text-mining', 'Co-expression', 'Predicted'])
            })
    
    # Sort by confidence score
    return sorted(interactions, key=lambda x: x['confidence'], reverse=True)

def render_function_tab(tabs, protein_info):
    """Render the function tab content"""
    if not protein_info:
        return
        
    with tabs[3]:  # Function
        st.markdown("### Protein Function Analysis")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Display function information
            st.markdown("#### Known Function")
            function_text = protein_info.get('function', 'Function unknown')
            st.markdown(f"**Function**: {function_text}")
            
            st.markdown("#### Subcellular Localization")
            locations = protein_info.get('subcellular_locations', [])
            if locations:
                for loc in locations:
                    st.markdown(f"- {loc}")
            else:
                st.markdown("Subcellular location unknown")
                
            # Sequence analysis for functional prediction
            sequence = protein_info.get('sequence', '')
            if sequence:
                st.markdown("#### Secondary Structure Prediction")
                
                if st.button("Predict Secondary Structure"):
                    with st.spinner("Predicting secondary structure..."):
                        sec_structure = predict_secondary_structure(sequence)
                        
                        # Map structure symbols to descriptions
                        structure_map = {
                            'H': 'Helix',
                            'E': 'Sheet',
                            'T': 'Turn',
                            'C': 'Coil'
                        }
                        
                        # Count structure elements
                        struct_counts = {s: sec_structure.count(s) for s in set(sec_structure)}
                        total = len(sec_structure)
                        
                        # Create structure pie chart
                        labels = [structure_map[s] for s in struct_counts.keys()]
                        values = list(struct_counts.values())
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=.3
                        )])
                        fig.update_layout(
                            title="Predicted Secondary Structure Distribution",
                            paper_bgcolor=plot_bg_color,
                            plot_bgcolor=plot_bg_color,
                            font=dict(color=plot_font_color)
                        )
                        st.plotly_chart(fig)
                        
                        # Display first 100 amino acids with colored structure
                        st.markdown("#### Structure Visualization (first 100 aa)")
                        
                        html = "<div style='font-family: monospace; line-height: 2.0;'>"
                        colors = {'H': '#FF5733', 'E': '#33A8FF', 'T': '#33FF57', 'C': '#C8C8C8'}
                        
                        for i, (aa, struct) in enumerate(zip(sequence[:100], sec_structure[:100])):
                            if i > 0 and i % 10 == 0:
                                html += "<br>"
                            html += f"<span style='background-color: {colors[struct]};'>{aa}</span>"
                        
                        html += "</div><br><div>"
                        for struct, color in colors.items():
                            html += f"<span style='background-color: {color}; padding: 0 10px;'></span> {structure_map[struct]} &nbsp;&nbsp;"
                        html += "</div>"
                        
                        st.markdown(html, unsafe_allow_html=True)
        
        with col2:
            # Protein interactions
            st.markdown("#### Protein-Protein Interactions")
            
            if st.button("Predict Protein Interactions"):
                with st.spinner("Analyzing protein interaction network..."):
                    interactions = predict_protein_interactions(protein_info)
                    
                    if interactions:
                        # Create interaction network visualization
                        center_protein = protein_info.get('name', 'Unknown')
                        
                        # Create network nodes with positions
                        nodes = [{'id': center_protein, 'label': center_protein, 'size': 20}]
                        
                        # Position interactors in a circle around the center
                        n_interactors = len(interactions)
                        radius = 1
                        
                        for i, interaction in enumerate(interactions):
                            angle = (2 * pi * i) / n_interactors
                            x = radius * cos(angle)
                            y = radius * sin(angle)
                            
                            nodes.append({
                                'id': interaction['protein'],
                                'label': interaction['protein'],
                                'size': 10 + (interaction['confidence'] * 10),
                                'x': x,
                                'y': y
                            })
                        
                        # Create network visualization using Plotly
                        edge_traces = []
                        
                        for interaction in interactions:
                            # Draw line from center to each interactor
                            x0, y0 = 0, 0  # Center coordinates
                            
                            # Find coordinates of target node
                            target_node = next((n for n in nodes if n['id'] == interaction['protein']), None)
                            if target_node:
                                x1, y1 = target_node.get('x', 0), target_node.get('y', 0)
                                
                                trace = go.Scatter(
                                    x=[x0, x1],
                                    y=[y0, y1],
                                    mode='lines',
                                    line=dict(width=1 + (interaction['confidence'] * 3), color='#888'),
                                    hoverinfo='text',
                                    text=f"Confidence: {interaction['confidence']}<br>Evidence: {interaction['evidence']}",
                                    showlegend=False
                                )
                                edge_traces.append(trace)
                        
                        # Create node trace
                        node_trace = go.Scatter(
                            x=[node.get('x', 0) for node in nodes],
                            y=[node.get('y', 0) for node in nodes],
                            mode='markers+text',
                            text=[node['label'] for node in nodes],
                            textposition="top center",
                            marker=dict(
                                size=[node['size'] for node in nodes],
                                color=['#FF4136' if node['id'] == center_protein else '#3D9970' for node in nodes],
                                line=dict(width=2, color='#fff')
                            ),
                            hoverinfo='text',
                            showlegend=False
                        )
                        
                        # Create figure with all traces
                        fig = go.Figure(data=edge_traces + [node_trace])
                        
                        fig.update_layout(
                            title="Protein Interaction Network",
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            paper_bgcolor=plot_bg_color,
                            plot_bgcolor=plot_bg_color,
                            font=dict(color=plot_font_color)
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Display interaction details
                        st.markdown("#### Interaction Details")
                        
                        for interaction in interactions:
                            st.markdown(f"**{interaction['protein']}**")
                            st.markdown(f"Confidence: {interaction['confidence']}")
                            st.markdown(f"Evidence: {interaction['evidence']}")
                            st.markdown("---")
                    else:
                        st.info("No interactions predicted for this protein.")
                        
            # Conserved regions
            st.markdown("#### Conserved Regions")
            
            if st.button("Analyze Conservation"):
                sequence = protein_info.get('sequence', '')
                if sequence:
                    with st.spinner("Analyzing sequence conservation..."):
                        conservation = analyze_conservation(sequence)
                        
                        if conservation:
                            # Create conservation bar chart
                            fig = go.Figure(data=[go.Bar(
                                x=list(conservation.keys()),
                                y=list(conservation.values())
                            )])
                            fig.update_layout(
                                title="Amino Acid Group Conservation",
                                paper_bgcolor=plot_bg_color,
                                plot_bgcolor=plot_bg_color,
                                font=dict(color=plot_font_color)
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("Conservation analysis failed.")
                else:
                    st.warning("Protein sequence not available for analysis.")

def render_comparison_tab(tabs, protein_info):
    """Render the comparison tab content"""
    if not protein_info:
        return
        
    with tabs[4]:  # Comparison
        st.markdown("### Protein Comparison")
        
        # Search for a protein to compare
        st.markdown("#### Select Protein for Comparison")
        comparison_protein = st.text_input("Enter protein name for comparison", "")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Current protein:** {protein_info.get('name', 'Unknown')}")
            
            # Display current protein info
            st.markdown("**Organism:** " + protein_info.get('organism', 'Unknown'))
            st.markdown("**Length:** " + str(protein_info.get('length', 'Unknown')))
            st.markdown("**Function:** " + protein_info.get('function', 'Unknown'))
            
            # Show current protein sequence (shortened)
            sequence = protein_info.get('sequence', '')
            if sequence:
                st.markdown("**Sequence preview:**")
                st.code(sequence[:50] + "..." if len(sequence) > 50 else sequence)
                
                # Basic properties
                properties = calculate_protein_properties(sequence)
                if properties:
                    st.markdown(f"**Molecular weight:** {properties['molecular_weight']:.1f} Da")
                    st.markdown(f"**Hydrophobic content:** {properties['hydrophobic_ratio'] * 100:.1f}%")
                    st.markdown(f"**Charged content:** {properties['charged_ratio'] * 100:.1f}%")
        
        # Only process comparison if a name is entered
        if comparison_protein:
            with col2:
                with st.spinner("Fetching comparison protein..."):
                    comparison_info = get_protein_info(comparison_protein)
                    
                    if comparison_info:
                        st.markdown(f"**Comparison protein:** {comparison_info.get('name', 'Unknown')}")
                        
                        # Display comparison protein info
                        st.markdown("**Organism:** " + comparison_info.get('organism', 'Unknown'))
                        st.markdown("**Length:** " + str(comparison_info.get('length', 'Unknown')))
                        st.markdown("**Function:** " + comparison_info.get('function', 'Unknown'))
                        
                        # Show comparison protein sequence (shortened)
                        comp_sequence = comparison_info.get('sequence', '')
                        if comp_sequence:
                            st.markdown("**Sequence preview:**")
                            st.code(comp_sequence[:50] + "..." if len(comp_sequence) > 50 else comp_sequence)
                            
                            # Basic properties
                            comp_properties = calculate_protein_properties(comp_sequence)
                            if comp_properties:
                                st.markdown(f"**Molecular weight:** {comp_properties['molecular_weight']:.1f} Da")
                                st.markdown(f"**Hydrophobic content:** {comp_properties['hydrophobic_ratio'] * 100:.1f}%")
                                st.markdown(f"**Charged content:** {comp_properties['charged_ratio'] * 100:.1f}%")
                    else:
                        st.warning(f"Could not find information for {comparison_protein}")
            
            # If both proteins are available, show comparison analysis
            if comparison_protein and comparison_info:
                st.markdown("### Comparative Analysis")
                
                sequence1 = protein_info.get('sequence', '')
                sequence2 = comparison_info.get('sequence', '')
                
                if sequence1 and sequence2:
                    if st.button("Compare Properties"):
                        with st.spinner("Analyzing properties..."):
                            # Get properties for both proteins
                            prop1 = calculate_protein_properties(sequence1)
                            prop2 = calculate_protein_properties(sequence2)
                            
                            if prop1 and prop2:
                                # Create comparative bar chart
                                categories = ['Length', 'Molecular Weight (kDa)', 'Hydrophobic %', 'Polar %', 'Charged %']
                                values1 = [
                                    prop1['length'], 
                                    prop1['molecular_weight']/1000, 
                                    prop1['hydrophobic_ratio']*100, 
                                    prop1['polar_ratio']*100, 
                                    prop1['charged_ratio']*100
                                ]
                                values2 = [
                                    prop2['length'], 
                                    prop2['molecular_weight']/1000, 
                                    prop2['hydrophobic_ratio']*100, 
                                    prop2['polar_ratio']*100, 
                                    prop2['charged_ratio']*100
                                ]
                                
                                fig = go.Figure(data=[
                                    go.Bar(name=protein_info.get('name', 'Protein 1'), x=categories, y=values1),
                                    go.Bar(name=comparison_info.get('name', 'Protein 2'), x=categories, y=values2)
                                ])
                                
                                fig.update_layout(
                                    title="Property Comparison",
                                    barmode='group',
                                    paper_bgcolor=plot_bg_color,
                                    plot_bgcolor=plot_bg_color,
                                    font=dict(color=plot_font_color)
                                )
                                st.plotly_chart(fig)
                                
                                # Calculate similarity score based on properties
                                similarity = 0
                                similarity += 1 - min(abs(prop1['hydrophobic_ratio'] - prop2['hydrophobic_ratio']) / max(prop1['hydrophobic_ratio'], prop2['hydrophobic_ratio']), 1)
                                similarity += 1 - min(abs(prop1['polar_ratio'] - prop2['polar_ratio']) / max(prop1['polar_ratio'], prop2['polar_ratio']), 1)
                                similarity += 1 - min(abs(prop1['charged_ratio'] - prop2['charged_ratio']) / max(prop1['charged_ratio'], prop2['charged_ratio']), 1)
                                similarity = (similarity / 3) * 100  # Convert to percentage
                                
                                # Display similarity score
                                st.markdown(f"### Property Similarity Score: {similarity:.1f}%")
                                
                                # Similarity gauge
                                fig = go.Figure(go.Indicator(
                                    mode = "gauge+number",
                                    value = similarity,
                                    title = {'text': "Similarity Score"},
                                    gauge = {
                                        'axis': {'range': [None, 100]},
                                        'bar': {'color': "green" if similarity > 70 else "orange" if similarity > 40 else "red"},
                                        'steps': [
                                            {'range': [0, 40], 'color': "lightgray"},
                                            {'range': [40, 70], 'color': "gray"},
                                            {'range': [70, 100], 'color': "darkgray"}
                                        ]
                                    }
                                ))
                                fig.update_layout(
                                    paper_bgcolor=plot_bg_color,
                                    font=dict(color=plot_font_color)
                                )
                                st.plotly_chart(fig)
                            else:
                                st.warning("Could not calculate properties for comparison.")
                    
                    if st.button("Compare Composition"):
                        with st.spinner("Analyzing composition..."):
                            # Count amino acids in both sequences
                            aa_count1 = {aa: sequence1.count(aa) for aa in set(sequence1)}
                            aa_count2 = {aa: sequence2.count(aa) for aa in set(sequence2)}
                            
                            # Calculate percentages
                            total1 = len(sequence1)
                            total2 = len(sequence2)
                            aa_percentage1 = {aa: (count / total1) * 100 for aa, count in aa_count1.items()}
                            aa_percentage2 = {aa: (count / total2) * 100 for aa, count in aa_count2.items()}
                            
                            # Get union of all amino acids
                            all_aas = sorted(set(sequence1) | set(sequence2))
                            
                            # Create dataframe for comparison
                            comp_data = {
                                'Amino Acid': all_aas,
                                f'{protein_info.get("name", "Protein 1")} (%)': [aa_percentage1.get(aa, 0) for aa in all_aas],
                                f'{comparison_info.get("name", "Protein 2")} (%)': [aa_percentage2.get(aa, 0) for aa in all_aas]
                            }
                            
                            comp_df = pd.DataFrame(comp_data)
                            st.dataframe(comp_df)
                            
                            # Create radar chart for composition comparison
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatterpolar(
                                r=[aa_percentage1.get(aa, 0) for aa in all_aas],
                                theta=all_aas,
                                fill='toself',
                                name=protein_info.get('name', 'Protein 1')
                            ))
                            
                            fig.add_trace(go.Scatterpolar(
                                r=[aa_percentage2.get(aa, 0) for aa in all_aas],
                                theta=all_aas,
                                fill='toself',
                                name=comparison_info.get('name', 'Protein 2')
                            ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, max(max(aa_percentage1.values()), max(aa_percentage2.values())) * 1.1]
                                    )),
                                title="Amino Acid Composition Comparison",
                                paper_bgcolor=plot_bg_color,
                                plot_bgcolor=plot_bg_color,
                                font=dict(color=plot_font_color)
                            )
                            st.plotly_chart(fig)
                else:
                    st.warning("Sequence data not available for comparison.")

# Main app functionality
def main():
    # Display app header
    st.markdown("# 🧬 Advanced Protein Explorer Pro")
    st.markdown("Comprehensive protein analysis and visualization platform for biologists and researchers")
    
    # Create columns for search and daily recommendations
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search bar with auto-suggestions
        protein_name = st.text_input("Search for a protein", st.session_state.last_search)
        st.markdown("<small>Examples: Hemoglobin, Insulin, Cytochrome C, Collagen, Albumin</small>", unsafe_allow_html=True)
    
    with col2:
        # Daily protein recommendation
        daily_protein = get_daily_protein()
        st.markdown(f"**📌 Today's Pick**: [**{daily_protein}**]")
        if st.button(f"Explore {daily_protein}"):
            protein_name = daily_protein
    
    # Process search when a protein name is entered
    if protein_name:
        st.session_state.last_search = protein_name  # Store last search
        
        with st.spinner(f"Searching for {protein_name}..."):
            protein_info = get_protein_info(protein_name)
            
            if protein_info:
                # Store protein info in session state
                st.session_state.protein_info = protein_info
                
                # Display overview information
                st.markdown(f"## {protein_info.get('name', 'Unknown Protein')}")
                
                col3, col4, col5 = st.columns([1, 1, 1])
                with col3:
                    st.markdown(f"**Organism**: {protein_info.get('organism', 'Unknown')}")
                with col4:
                    st.markdown(f"**Gene**: {protein_info.get('gene', 'Unknown')}")
                with col5:
                    st.markdown(f"**Length**: {protein_info.get('length', 'Unknown')} amino acids")
                
                # Create tabs for detailed information
                tabs = st.tabs(["Overview", "Composition", "Structure", "Function", "Comparison"])
                
                # Overview tab
                with tabs[0]:
                    st.markdown("### Protein Overview")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### Description")
                        st.markdown(protein_info.get('function', 'No description available.'))
                        
                        st.markdown("#### Sequence")
                        sequence = protein_info.get('sequence', 'Sequence not available')
                        
                        if len(sequence) > 100:
                            st.code(sequence[:100] + "...", language=None)
                            st.markdown(f"**Total length:** {len(sequence)} amino acids")
                        else:
                            st.code(sequence, language=None)
                        
                        # Download sequence as FASTA
                        if sequence != 'Sequence not available':
                            fasta_file = download_fasta(protein_info.get('name', 'protein'), sequence)
                            st.download_button(
                                "📥 Download FASTA",
                                fasta_file,
                                file_name=f"{protein_name.replace(' ', '_')}.fasta",
                                mime="text/plain"
                            )
                    
                    with col2:
                        st.markdown("#### Key Properties")
                        
                        # Calculate basic properties
                        if sequence != 'Sequence not available':
                            properties = calculate_protein_properties(sequence)
                            if properties:
                                st.markdown(f"**Molecular weight:** {properties['molecular_weight']:.1f} Da")
                                st.markdown(f"**Hydrophobic residues:** {properties['hydrophobic_ratio'] * 100:.1f}%")
                                st.markdown(f"**Charged residues:** {properties['charged_ratio'] * 100:.1f}%")
                                st.markdown(f"**Polar residues:** {properties['polar_ratio'] * 100:.1f}%")
                        
                        # Display additional protein information
                        st.markdown("#### Additional Information")
                        st.markdown(f"**UniProt entry:** [Link to UniProt](https://www.uniprot.org/uniprotkb?query={protein_name})")
                        
                        if protein_info.get('pdb_id'):
                            st.markdown(f"**PDB entry:** [Link to PDB](https://www.rcsb.org/structure/{protein_info['pdb_id']})")
                
                # Render other tabs
                render_composition_tab(tabs, protein_info)
                render_structure_tab(tabs, protein_info)
                render_function_tab(tabs, protein_info)
                render_comparison_tab(tabs, protein_info)
                
            else:
                st.error(f"Could not find information for {protein_name}. Please check the spelling or try another protein.")
    else:
        # Display welcome page if no search has been made
        st.markdown("## Welcome to Advanced Protein Explorer Pro!")
        st.markdown("""
        This tool allows researchers to analyze and visualize protein data with advanced features:
        
        - **Composition Analysis**: Amino acid distribution, hydropathy plots, and physicochemical properties
        - **Structure Visualization**: 3D structure viewing, domain analysis, and secondary structure prediction
        - **Function Analysis**: Interaction network mapping and binding site prediction
        - **Protein Comparison**: Compare properties and composition between different proteins
        
        Get started by searching for a protein above!
        """)
        
        # Display featured proteins
        st.markdown("### Featured Proteins")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Hemoglobin"):
                st.session_state.last_search = "Hemoglobin"
                st.experimental_rerun()
                
        with col2:
            if st.button("Insulin"):
                st.session_state.last_search = "Insulin"
                st.experimental_rerun()
                
        with col3:
            if st.button("Collagen"):
                st.session_state.last_search = "Collagen"
                st.experimental_rerun()
                
        with col4:
            if st.button("Cytochrome C"):
                st.session_state.last_search = "Cytochrome C"
                st.experimental_rerun()

if __name__ == "__main__":
    main()