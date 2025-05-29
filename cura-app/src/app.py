# /src/app.py
import os
import re
import sys
import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config

# Imports for Chatbot
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

# --- Step 1: Set up the environment by loading secrets ---
# This robustly finds the .env file by navigating from this script's location.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, 'secrets', 'prod.env')

if os.path.exists(ENV_FILE_PATH):
    load_dotenv(ENV_FILE_PATH)
else:
    # This handles deployment environments where secrets are set directly.
    print("Running in deployed environment or 'secrets/prod.env' not found. Relying on preset environment variables.")


# --- Step 2: Import your application logic ---
# Now that the environment is set, we can import our RAG processor.
try:
    from core.rag_processor import get_regulation_summary 
    RAG_FUNCTION_AVAILABLE = True
except ImportError as e:
    RAG_FUNCTION_AVAILABLE = False
    st.error(f"Could not import RAG processor. Ensure 'rag_processor.py' is in 'src/core/'. Error: {e}")


# --- Step 3: Configure the Streamlit App ---
st.set_page_config(layout="wide")
st.title("Chemical Unified Regulatory Atlas (CURA)")


# --- Step 4: Load credentials from environment variables ---
# The .env file has already been loaded; now we retrieve the values.
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER") 
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") # For backwards compatibility if keys differ
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# LLM Provider Variables
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 

# Main app layout with tabs
tab1, tab2 = st.tabs(["Graph Explorer", "Chatbot"])


# --- TAB 1: GRAPH EXPLORER ---
with tab1:
    with st.sidebar:
        st.title("CURA Explore Search")
        search_query_graph = st.text_input("Search Chemical (Name or CAS) for Graph:", help="e.g., Benzene or 71-43-2", key="graph_search_input")
        st.markdown("---") 
        st.subheader("Graph Regulation Summary")
        summary_placeholder_graph = st.empty()
        summary_placeholder_graph.info("Enter a search term above to see regulations for the graph.")
        
        rag_summary_container = st.container()

    st.header("CURA Explorer")

    @st.cache_resource
    def get_graph_explorer_driver():
        # Use NEO4J_USER if available, otherwise fall back to NEO4J_USERNAME
        current_neo4j_user = NEO4J_USER if NEO4J_USER else NEO4J_USERNAME
        if not NEO4J_URI or not current_neo4j_user or not NEO4J_PASSWORD:
            st.error("Neo4j credentials not found. Please check your 'secrets/prod.env' file or environment variables.")
            return None
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(current_neo4j_user, NEO4J_PASSWORD))
            driver.verify_connectivity()
            return driver
        except Exception as e:
            st.error(f"Failed to connect to Neo4j for Graph Explorer: {e}")
            return None

    graph_driver = get_graph_explorer_driver()

    # --- Graph Data Fetching and Visualization Functions (No changes needed here) ---
    def add_node_if_not_exists(node_list, existing_ids_set, node_id, label, title, shape, color, size):
        if node_id not in existing_ids_set:
            node_list.append(Node(id=node_id, label=label, title=title, shape=shape, color=color, size=size))
            existing_ids_set.add(node_id)

    def add_edge_if_not_exists(edge_list, existing_edge_tuples_set, source, target, label):
        edge_tuple = (source, target, label)
        if edge_tuple not in existing_edge_tuples_set:
            edge_list.append(Edge(source=source, target=target, label=label))
            existing_edge_tuples_set.add(edge_tuple)

    def fetch_graph_data(driver_instance, search_term):
        if not driver_instance or not search_term:
            return [], [], [], None, None, [] 
        nodes_viz, edges_viz, processed_node_ids, processed_edge_tuples = [], [], set(), set()
        regulations_summary, all_names_for_found_cas = set(), set()
        found_cas, found_chem_name = None, None
        is_cas_search = bool(re.match(r"^\d{2,7}-\d{2}-\d$", search_term.strip()))

        with driver_instance.session() as session:
            # The comprehensive graph fetching logic remains the same.
            if is_cas_search:
                cas_to_search = search_term.strip()
                chem_records = session.run("MATCH (c:Chemical {cas: $cas}) RETURN c", cas=cas_to_search)
                chem_record = chem_records.single()
                if chem_record and chem_record["c"]:
                    c_props = chem_record["c"]
                    found_cas = c_props["cas"]
                    add_node_if_not_exists(nodes_viz, processed_node_ids, found_cas, f"CAS: {found_cas}", f"Chemical: {found_cas}", "ellipse", "#FFADAD", 25)
                    name_records = session.run("MATCH (cn:ChemicalName)-[:IS_NAME_OF]->(c:Chemical {cas: $cas}) RETURN cn", cas=found_cas)
                    for i, rec in enumerate(name_records):
                        cn_props = rec["cn"]
                        chem_name_val = cn_props["name"]
                        all_names_for_found_cas.add(chem_name_val)
                        if i == 0 and not found_chem_name: found_chem_name = chem_name_val
                        add_node_if_not_exists(nodes_viz, processed_node_ids, chem_name_val, chem_name_val, f"Name: {chem_name_val}", "box", "#ADD8E6", 15)
                        add_edge_if_not_exists(edges_viz, processed_edge_tuples, chem_name_val, found_cas, "IS_NAME_OF")
                        name_reg_records = session.run("MATCH (cn:ChemicalName {name: $name_val})-[:IS_REGULATED]->(r:Regulation) RETURN r", name_val=chem_name_val)
                        for reg_rec in name_reg_records:
                            reg_props = reg_rec["r"]
                            reg_name = reg_props["name"]
                            regulations_summary.add(reg_name)
                            add_node_if_not_exists(nodes_viz, processed_node_ids, reg_name, reg_name, f"Regulation: {reg_name}", "hexagon", "#90EE90", 20)
                            add_edge_if_not_exists(edges_viz, processed_edge_tuples, chem_name_val, reg_name, "IS_REGULATED")
                    cas_reg_records = session.run("MATCH (c:Chemical {cas: $cas})-[:IS_REGULATED]->(r:Regulation) RETURN r", cas=found_cas)
                    for rec in cas_reg_records:
                        reg_props = rec["r"]
                        reg_name = reg_props["name"]
                        regulations_summary.add(reg_name)
                        add_node_if_not_exists(nodes_viz, processed_node_ids, reg_name, reg_name, f"Regulation: {reg_name}", "hexagon", "#90EE90", 20)
                        add_edge_if_not_exists(edges_viz, processed_edge_tuples, found_cas, reg_name, "IS_REGULATED")
            else: # Search by Name
                name_to_search = search_term.strip()
                cn_records = session.run("MATCH (cn:ChemicalName) WHERE toLower(cn.name) CONTAINS toLower($name_query) RETURN cn LIMIT 1", name_query=name_to_search)
                cn_record = cn_records.single()
                if cn_record and cn_record["cn"]:
                    cn_props = cn_record["cn"]
                    found_chem_name = cn_props["name"]
                    add_node_if_not_exists(nodes_viz, processed_node_ids, found_chem_name, found_chem_name, f"Name: {found_chem_name}", "box", "#ADD8E6", 15)
                    cas_records = session.run("MATCH (cn:ChemicalName {name: $name})-[:IS_NAME_OF]->(c:Chemical) RETURN c", name=found_chem_name)
                    cas_record = cas_records.single()
                    if cas_record and cas_record["c"]:
                        c_props = cas_record["c"]
                        found_cas = c_props["cas"]
                        add_node_if_not_exists(nodes_viz, processed_node_ids, found_cas, f"CAS: {found_cas}", f"Chemical: {found_cas}", "ellipse", "#FFADAD", 25)
                        add_edge_if_not_exists(edges_viz, processed_edge_tuples, found_chem_name, found_cas, "IS_NAME_OF")
                        all_names_records_for_cas = session.run("MATCH (cname:ChemicalName)-[:IS_NAME_OF]->(c:Chemical {cas: $cas}) RETURN cname.name", cas=found_cas)
                        for rec_name in all_names_records_for_cas: all_names_for_found_cas.add(rec_name["cname.name"])
                        cas_reg_records = session.run("MATCH (c:Chemical {cas: $cas})-[:IS_REGULATED]->(r:Regulation) RETURN r", cas=found_cas)
                        for rec in cas_reg_records:
                            reg_props = rec["r"]; reg_name = reg_props["name"]; regulations_summary.add(reg_name)
                            add_node_if_not_exists(nodes_viz, processed_node_ids, reg_name, reg_name, f"Regulation: {reg_name}", "hexagon", "#90EE90", 20)
                            add_edge_if_not_exists(edges_viz, processed_edge_tuples, found_cas, reg_name, "IS_REGULATED")
                    name_reg_records = session.run("MATCH (cn:ChemicalName {name: $name})-[:IS_REGULATED]->(r:Regulation) RETURN r", name=found_chem_name)
                    for rec in name_reg_records:
                        reg_props = rec["r"]; reg_name = reg_props["name"]; regulations_summary.add(reg_name)
                        add_node_if_not_exists(nodes_viz, processed_node_ids, reg_name, reg_name, f"Regulation: {reg_name}", "hexagon", "#90EE90", 20)
                        add_edge_if_not_exists(edges_viz, processed_edge_tuples, found_chem_name, reg_name, "IS_REGULATED")
            if found_cas and not found_chem_name:
                name_result = session.run("MATCH (cn:ChemicalName)-[:IS_NAME_OF]->(c:Chemical {cas: $cas}) RETURN cn.name", cas=found_cas)
                first_name_found = False
                for rec in name_result:
                    all_names_for_found_cas.add(rec["cn.name"])
                    if not first_name_found: found_chem_name = rec["cn.name"]; first_name_found = True
        return nodes_viz, edges_viz, sorted(list(regulations_summary)), found_cas, found_chem_name, sorted(list(all_names_for_found_cas))

    # --- Main Application Logic ---
    if not graph_driver:
        st.warning("Graph Explorer: Neo4j driver not available.")
    else:
        graph_viz_placeholder = st.empty() 
        if search_query_graph: 
            nodes, edges, regulations, display_cas, display_name, all_names = fetch_graph_data(graph_driver, search_query_graph)
            
            summary_lines = []
            if display_cas or display_name:
                summary_title_text = f"**{display_name}** (CAS: {display_cas})" if display_name and display_cas else f"**{display_name}**" if display_name else f"CAS: **{display_cas}**"
                if summary_title_text: summary_lines.append(summary_title_text)
                
                other_names_to_show = [name for name in all_names if name != display_name] or (all_names if not display_name else [])
                if other_names_to_show: summary_lines.append(f"\nAlso known as: {', '.join(other_names_to_show)}")
                
                if regulations:
                    summary_lines.append(f"\nIs regulated in:")
                    for reg in regulations: summary_lines.append(f"- {reg}")
                elif summary_lines: summary_lines.append("\nNot found in any specific regulations in the graph.")
                
                if summary_lines: summary_placeholder_graph.markdown("\n".join(summary_lines))
                else: summary_placeholder_graph.warning(f"Details for '{search_query_graph}' found, but no regulations linked.")
            else:
                summary_placeholder_graph.error(f"No chemical found for '{search_query_graph}' in graph.")

            with rag_summary_container: 
                if regulations and (display_name or display_cas):
                    st.markdown("---")
                    st.subheader("Detailed Document Summary (RAG)")
                    
                    selected_regulation_for_rag = st.selectbox(
                        "Select a regulation for detailed summary:",
                        options=[""] + regulations, 
                        key=f"rag_select_{search_query_graph}"
                    )

                    if selected_regulation_for_rag:
                        chemical_identifier_for_rag = display_name or display_cas or (all_names[0] if all_names else None)
                        if chemical_identifier_for_rag:
                            st.markdown(f"Generating summary for **{chemical_identifier_for_rag}** in **{selected_regulation_for_rag}**...")
                            if RAG_FUNCTION_AVAILABLE:
                                with st.spinner(f"Fetching summary from '{selected_regulation_for_rag}'..."):
                                    summary_text = get_regulation_summary(selected_regulation_for_rag, chemical_identifier_for_rag)
                                    st.success("Summary Generated:")
                                    st.info(summary_text)
                            else:
                                st.error("RAG summarization function is not available. Check import and file location.")
                        else:
                            st.warning("Could not determine chemical identifier for RAG summary.")
                elif search_query_graph: 
                     st.markdown("---")
                     st.warning("No regulations found for this chemical to generate a detailed summary.")

            if nodes:
                config = Config(width="100%", height=700, directed=True, physics=False, hierarchical=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
                with graph_viz_placeholder.container(): agraph(nodes=nodes, edges=edges, config=config)
            else:
                with graph_viz_placeholder.container(): st.info("No graph to display for the current search.")
        else:
            with graph_viz_placeholder.container(): st.info("Enter a chemical name or CAS number in the sidebar to visualize its relationships.")


# --- TAB 2: CHATBOT ---
with tab2:
    st.header("CURA Chat")
    
    # Check for Neo4j credentials first
    if not NEO4J_URI or not (NEO4J_USERNAME or NEO4J_USER) or not NEO4J_PASSWORD:
        st.error("Chatbot: Neo4j credentials not found. Check your 'secrets/prod.env' file.")
    # Check for at least one LLM provider
    elif not GOOGLE_API_KEY and not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME]):
        st.error("Chatbot: LLM provider credentials not found. Please provide either GOOGLE_API_KEY or all AZURE_OPENAI_* variables in your 'secrets/prod.env' file.")
    else:
        try:
            # --- Chatbot LLM Selection ---
            chatbot_llm = None
            if all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME]):
                st.info("Chatbot is using Azure OpenAI.")
                chatbot_llm = AzureChatOpenAI(
                    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                    temperature=0
                )
            elif GOOGLE_API_KEY:
                st.info("Chatbot is using Google Gemini.")
                chatbot_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0
                )

            chatbot_neo4j_user = NEO4J_USERNAME if NEO4J_USERNAME else NEO4J_USER
            graph_for_chatbot = Neo4jGraph(url=NEO4J_URI, username=chatbot_neo4j_user, password=NEO4J_PASSWORD)
            graph_for_chatbot.query("RETURN 1") 
            
            chain = GraphCypherQAChain.from_llm(
                chatbot_llm, 
                graph=graph_for_chatbot, 
                verbose=True,
                allow_dangerous_requests=True 
            )

            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "Hi, how can I help you with the chemical graph data?"}]

            for msg in st.session_state.messages: 
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input(placeholder="e.g., What chemicals are regulated by Apple?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                    response = chain.run(st.session_state.messages[-1]["content"], callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)

        except Exception as e:
            st.error(f"Chatbot: Failed to initialize or connect. Error: {e}")