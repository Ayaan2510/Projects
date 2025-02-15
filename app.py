import streamlit as st
import requests
import json
import plotly.graph_objects as go
import pandas as pd
import time
import random
from io import BytesIO
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="🧬 Advanced Protein Explorer Pro",
    page_icon="🧬",
    layout="wide"
)

# List of random proteins for daily cycling
PROTEIN_LIST = ["Hemoglobin", "Insulin", "Myoglobin", "Collagen", "Ferritin", "Cytochrome C", "Albumin", "Keratin"]

def get_daily_protein():
    """Selects a daily protein based on the current date"""
    random.seed(datetime.now().date().toordinal())
    return random.choice(PROTEIN_LIST)

# Custom CSS with dark mode toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("""
    <style>
    .main { background-color: #121212; color: white; }
    .stButton>button { background-color: #4a69bd; color: white; }
    h1, h2, h3 { color: #1e3799; }
    .stTextInput>div>div>input { border: 2px solid #4a69bd; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def get_protein_info(protein_name):
    """Fetch detailed protein information from UniProt API and cache results"""
    try:
        search_url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}&format=json"
        response = requests.get(search_url)
        data = response.json()
        
        if 'results' in data and data['results']:
            result = data['results'][0]
            return {
                'name': result.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown'),
                'function': next((c.get('value') for c in result.get('comments', []) if c.get('type') == 'FUNCTION'), 'Function unknown'),
                'gene': result.get('genes', [{}])[0].get('geneName', {}).get('value', 'Unknown'),
                'organism': result.get('organism', {}).get('scientificName', 'Unknown'),
                'length': result.get('sequence', {}).get('length', 'Unknown'),
                'mass': result.get('sequence', {}).get('mass', 'Unknown'),
                'subcellular_locations': [loc.get('value') for loc in result.get('subcellularLocations', [])],
                'sequence': result.get('sequence', {}).get('value', 'Sequence not available'),
                'pdb_id': next((xref.get('id') for xref in result.get('uniProtKBCrossReferences', []) if xref.get('database') == 'PDB'), None)
            }
        return None
    except Exception as e:
        st.error(f"Error fetching protein information: {str(e)}")
        return None

def plot_amino_acid_composition(sequence):
    """Plot amino acid composition of the protein"""
    if not sequence:
        return None
    aa_count = {aa: sequence.count(aa) for aa in set(sequence)}
    aa_sorted = sorted(aa_count.items(), key=lambda x: x[1], reverse=True)
    
    fig = go.Figure(data=[go.Bar(x=[aa[0] for aa in aa_sorted], y=[aa[1] for aa in aa_sorted],
                                 text=[aa[1] for aa in aa_sorted], textposition='auto')])
    fig.update_layout(title="Amino Acid Composition", xaxis_title="Amino Acid", yaxis_title="Count")
    return fig

def download_fasta(protein_name, sequence):
    """Download protein sequence in FASTA format"""
    fasta_format = f">{protein_name}\n{sequence}"
    return BytesIO(fasta_format.encode())

def main():
    st.title("🧬 Advanced Protein Explorer Pro")
    st.markdown("### Discover the intricate world of proteins and their functions")
    
    daily_protein = get_daily_protein()
    st.sidebar.markdown(f"**🔬 Today's Protein: {daily_protein}**")
    
    # Add a dropdown for quick protein selection
    quick_select = st.sidebar.selectbox("Quick Select Protein:", [""] + PROTEIN_LIST)
    
    protein_name = st.text_input("Enter a protein name or UniProt ID:", value=quick_select, placeholder=f"e.g., '{daily_protein}'")
    analyze_button = st.button("🔍 Analyze Protein")
    
    if analyze_button or protein_name:
        selected_protein = protein_name if protein_name else daily_protein
        with st.spinner(f"Fetching data for {selected_protein}..."):
            time.sleep(2)
            protein_info = get_protein_info(selected_protein)
            
            if protein_info:
                # Create tabs for different sections
                tabs = st.tabs(["Overview", "Sequence", "Structure", "Interactions", "Analysis"])
                
                with tabs[0]:  # Overview
                    st.markdown(f"""
                    ## Protein Information
                    - **Name:** {protein_info['name']}
                    - **Gene:** {protein_info['gene']}
                    - **Organism:** {protein_info['organism']}
                    - **Length:** {protein_info['length']} amino acids
                    - **Mass:** {protein_info['mass']} Da
                    - **Function:** {protein_info['function']}
                    - **Subcellular Locations:** {', '.join(protein_info['subcellular_locations']) if protein_info['subcellular_locations'] else 'Not specified'}
                    """)
                
                with tabs[1]:  # Sequence
                    st.markdown("### Protein Sequence")
                    st.code(protein_info['sequence'])
                    fasta_file = download_fasta(protein_info['name'], protein_info['sequence'])
                    st.download_button("📥 Download FASTA", fasta_file, file_name=f"{selected_protein}.fasta", mime="text/plain")
                    
                    # Add sequence analysis options
                    analysis_options = st.multiselect("Choose sequence analysis:", ["Amino Acid Composition", "Hydrophobicity Plot", "Secondary Structure Prediction"])
                    if "Amino Acid Composition" in analysis_options:
                        st.plotly_chart(plot_amino_acid_composition(protein_info['sequence']))
                    # Implement other analysis options as needed
                
                with tabs[2]:  # Structure
                    st.markdown("### Protein Structure")
                    if protein_info['pdb_id']:
                        pdb_id = protein_info['pdb_id']
                        st.image(f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_assembly-1.jpeg",
                                 caption=f"Structure of {protein_info['name']} (PDB ID: {pdb_id})")
                        st.markdown(f"[View 3D Structure on PDB](https://www.rcsb.org/3d-view/{pdb_id})")
                    else:
                        st.info("No structure available for this protein in the PDB.")
                    
                    # Add structure analysis options
                    st.markdown("### Structure Analysis Tools")
                    st.button("Calculate Surface Area")
                    st.button("Identify Binding Sites")
                    st.button("Analyze Structural Domains")
                
                with tabs[3]:  # Interactions
                    st.markdown("### Protein Interactions")
                    string_url = f"https://string-db.org/network/{protein_info['gene']}"
                    st.markdown(f"[🔗 View Protein Interactions on STRING]({string_url})")
                    
                    # Add interaction analysis options
                    st.markdown("### Interaction Analysis")
                    st.button("Predict Protein-Protein Interactions")
                    st.button("Analyze Interaction Network")
                
                with tabs[4]:  # Analysis
                    st.markdown("### Advanced Analysis Tools")
                    st.selectbox("Choose Analysis Tool:", ["Evolutionary Conservation", "Mutation Impact Prediction", "Protein Family Classification"])
                    st.button("Run Analysis")
            else:
                st.error("No protein data found. Please check the name or ID and try again.")
    
    # Add a feedback section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Feedback")
    feedback = st.sidebar.text_area("Share your thoughts or report issues:")
    if st.sidebar.button("Submit Feedback"):
        st.sidebar.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
