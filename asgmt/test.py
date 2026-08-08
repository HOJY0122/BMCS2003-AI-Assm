import streamlit as st
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Student Mental Health Predictor", layout="wide")

st.title("Student Mental Health Prediction System")
st.write("Predict likelihood of depression using the custom Decision Tree flow.")

# --- Helper Functions ---
def parse_year(val):
    if pd.isna(val):
        return 1
    val_str = str(val).lower()
    numbers = re.findall(r"\d+", val_str)
    return int(numbers[0]) if numbers else 1

def is_yes(val):
    return str(val).strip().lower() == "yes"

# --- Custom Decision Tree Logic ---
def predict_custom_tree(inputs):
    path_nodes = []
    
    # Root: Age >= 20 ?
    path_nodes.append("age >= 20 ?")
    if inputs['age'] >= 20:
        path_nodes.append("Year of Study >= 3 ? (R)")
        if inputs['year'] >= 3:
            path_nodes.append("Depress (R_Year)")
            return "Depress", path_nodes
        else:
            path_nodes.append("CGPA <= 2.0 ? (R)")
            if inputs['cgpa'] <= 2.0:
                path_nodes.append("Depress (R_CGPA)")
                return "Depress", path_nodes
            else:
                return traverse_health_subtree(inputs, path_nodes, prefix="R")
    else:
        path_nodes.append("CGPA <= 2.0 ? (L)")
        if inputs['cgpa'] <= 2.0:
            path_nodes.append("Depress (L_CGPA)")
            return "Depress", path_nodes
        else:
            return traverse_health_subtree(inputs, path_nodes, prefix="L")

def traverse_health_subtree(inputs, path_nodes, prefix="R"):
    path_nodes.append(f"Marital ? ({prefix})")
    if is_yes(inputs['marital']):
        path_nodes.append(f"Depress ({prefix}_Marital)")
        return "Depress", path_nodes
    
    path_nodes.append(f"Anxiety ? ({prefix})")
    if is_yes(inputs['anxiety']):
        path_nodes.append(f"Depress ({prefix}_Anxiety)")
        return "Depress", path_nodes
    
    path_nodes.append(f"Panic ? ({prefix})")
    if is_yes(inputs['panic']):
        path_nodes.append(f"Depress ({prefix}_Panic)")
        return "Depress", path_nodes
    
    path_nodes.append(f"Treatment ? ({prefix})")
    if is_yes(inputs['treatment']):
        path_nodes.append(f"Depress ({prefix}_Treatment)")
        return "Depress", path_nodes
    else:
        path_nodes.append(f"No Depress ({prefix}_Treatment)")
        return "No Depress", path_nodes

# --- Matplotlib Decision Tree Plotter ---
def render_matplotlib_tree(active_path):
    G = nx.DiGraph()

    # Exact coordinates matching network layout
    pos = {
        # Root
        "age >= 20 ?": (0, 7),
        
        # Left Branch (Age >= 20 False)
        "CGPA <= 2.0 ? (L)": (-4, 6),
        "Depress (L_CGPA)": (-2, 5),
        "Marital ? (L)": (-6, 5),
        "Depress (L_Marital)": (-4, 4),
        "Anxiety ? (L)": (-8, 4),
        "Depress (L_Anxiety)": (-6, 3),
        "Panic ? (L)": (-10, 3),
        "Depress (L_Panic)": (-8, 2),
        "Treatment ? (L)": (-12, 2),
        "Depress (L_Treatment)": (-10, 1),
        "No Depress (L_Treatment)": (-14, 1),

        # Right Branch (Age >= 20 True)
        "Year of Study >= 3 ? (R)": (4, 6),
        "Depress (R_Year)": (8, 5),
        "CGPA <= 2.0 ? (R)": (2, 5),
        "Depress (R_CGPA)": (4, 4),
        "Marital ? (R)": (0, 4),
        "Depress (R_Marital)": (2, 3),
        "Anxiety ? (R)": (-2, 3),
        "Depress (R_Anxiety)": (0, 2),
        "Panic ? (R)": (-4, 2),
        "Depress (R_Panic)": (-2, 1),
        "Treatment ? (R)": (-6, 1),
        "Depress (R_Treatment)": (-4, 0),
        "No Depress (R_Treatment)": (-8, 0)
    }

    # Display Labels
    labels = {
        "age >= 20 ?": "age >= 20 ?",
        "CGPA <= 2.0 ? (L)": "CGPA <= 2.0 ?",
        "Depress (L_CGPA)": "Depress",
        "Marital ? (L)": "Marital ?",
        "Depress (L_Marital)": "Depress",
        "Anxiety ? (L)": "Anxiety ?",
        "Depress (L_Anxiety)": "Depress",
        "Panic ? (L)": "Panic ?",
        "Depress (L_Panic)": "Depress",
        "Treatment ? (L)": "Threatment ?",
        "Depress (L_Treatment)": "Depress",
        "No Depress (L_Treatment)": "No Depress",

        "Year of Study >= 3 ? (R)": "Year of Study >= 3 ?",
        "Depress (R_Year)": "Depress",
        "CGPA <= 2.0 ? (R)": "CGPA <= 2.0 ?",
        "Depress (R_CGPA)": "Depress",
        "Marital ? (R)": "Marital ?",
        "Depress (R_Marital)": "Depress",
        "Anxiety ? (R)": "Anxiety ?",
        "Depress (R_Anxiety)": "Depress",
        "Panic ? (R)": "Panic ?",
        "Depress (R_Panic)": "Depress",
        "Treatment ? (R)": "Threatment ?",
        "Depress (R_Treatment)": "Depress",
        "No Depress (R_Treatment)": "No Depress"
    }

    edges = [
        # Root splits
        ("age >= 20 ?", "CGPA <= 2.0 ? (L)", "False"),
        ("age >= 20 ?", "Year of Study >= 3 ? (R)", "True"),

        # Left Subtree splits
        ("CGPA <= 2.0 ? (L)", "Depress (L_CGPA)", "True"),
        ("CGPA <= 2.0 ? (L)", "Marital ? (L)", "False"),
        ("Marital ? (L)", "Depress (L_Marital)", "True"),
        ("Marital ? (L)", "Anxiety ? (L)", "False"),
        ("Anxiety ? (L)", "Depress (L_Anxiety)", "True"),
        ("Anxiety ? (L)", "Panic ? (L)", "False"),
        ("Panic ? (L)", "Depress (L_Panic)", "True"),
        ("Panic ? (L)", "Treatment ? (L)", "False"),
        ("Treatment ? (L)", "Depress (L_Treatment)", "True"),
        ("Treatment ? (L)", "No Depress (L_Treatment)", "False"),

        # Right Subtree splits
        ("Year of Study >= 3 ? (R)", "Depress (R_Year)", "True"),
        ("Year of Study >= 3 ? (R)", "CGPA <= 2.0 ? (R)", "False"),
        ("CGPA <= 2.0 ? (R)", "Depress (R_CGPA)", "True"),
        ("CGPA <= 2.0 ? (R)", "Marital ? (R)", "False"),
        ("Marital ? (R)", "Depress (R_Marital)", "True"),
        ("Marital ? (R)", "Anxiety ? (R)", "False"),
        ("Anxiety ? (R)", "Depress (R_Anxiety)", "True"),
        ("Anxiety ? (R)", "Panic ? (R)", "False"),
        ("Panic ? (R)", "Depress (R_Panic)", "True"),
        ("Panic ? (R)", "Treatment ? (R)", "False"),
        ("Treatment ? (R)", "Depress (R_Treatment)", "True"),
        ("Treatment ? (R)", "No Depress (R_Treatment)", "False"),
    ]

    for u, v, w in edges:
        G.add_edge(u, v, label=w)

    fig, ax = plt.subplots(figsize=(18, 10))

    # Highlight nodes along active path
    node_border_colors = ['red' if node in active_path else 'black' for node in G.nodes()]
    node_widths = [3.0 if node in active_path else 1.0 for node in G.nodes()]

    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_shape='s',
        node_size=4200,
        node_color='white',
        edgecolors=node_border_colors,
        linewidths=node_widths,
        ax=ax
    )

    # Draw Labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family="sans-serif", ax=ax)

    # Highlight active path edges
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if u in active_path and v in active_path and active_path.index(v) == active_path.index(u) + 1:
            edge_colors.append('red')
            edge_widths.append(2.5)
        else:
            edge_colors.append('black')
            edge_widths.append(1.0)

    # Draw Edges
    nx.draw_networkx_edges(
        G, pos,
        edgelist=G.edges(),
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=15,
        ax=ax
    )

    # Draw Branch Labels (True / False)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='blue', ax=ax)

    plt.title("Decision Tree Flowchart (Red border highlights the active user path)", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    return fig

# --- User Input UI ---
st.header("Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=30, value=20)
    cgpa = st.number_input(
        "Enter CGPA (0.00 to 4.00)", 
        min_value=0.0, 
        max_value=4.0, 
        value=3.25, 
        step=0.01, 
        format="%.2f"
    )
    year_str = st.selectbox("Year of Study", ["Year 1", "Year 2", "Year 3", "Year 4"], index=0)
    marital = st.selectbox("Marital Status", ["No", "Yes"])

with col2:
    anxiety = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
    panic = st.selectbox("Do you have Panic attack?", ["No", "Yes"])
    treatment = st.selectbox("Did you seek Specialist Treatment?", ["No", "Yes"])

if st.button("Predict Mental Health Status", type="primary"):
    parsed_inputs = {
        'age': age,
        'cgpa': cgpa,
        'year': parse_year(year_str),
        'marital': marital,
        'anxiety': anxiety,
        'panic': panic,
        'treatment': treatment
    }

    prediction, path_taken = predict_custom_tree(parsed_inputs)

    if prediction == "Depress":
        st.error(f"Prediction Result: **{prediction.upper()}** (High indication of Depression)")
    else:
        st.success(f"Prediction Result: **{prediction.upper()}** (Low indication of Depression)")

    st.subheader("Decision Tree Model Flow")
    fig = render_matplotlib_tree(path_taken)
    st.pyplot(fig)