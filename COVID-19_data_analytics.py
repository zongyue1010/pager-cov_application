import streamlit as st
from streamlit_agraph import agraph, TripleStore, Config
import pandas as pd
import requests
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import generateheatmap as Heatmap
import generateNetwork as Network
import numpy as np
import plotly.tools
import base64
from streamlit_agraph import agraph, Node, Edge, Config
import plotly
import math

# color mapping of the gene expression #
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
colorUnit = 56
top = cm.get_cmap('Blues_r', colorUnit)
bottom = cm.get_cmap('Reds', colorUnit)
newcolors = np.vstack((
    top(np.linspace(0, 1, 56)),([[1,1,1,0]]),
    bottom(np.linspace(0, 1, 56))
))
newcmp = ListedColormap(newcolors, name='RedBlue')
hex_map = [matplotlib.colors.to_hex(i, keep_alpha=True) for i in newcolors]


#st.title('GBM-PDX Data Analysis in U01 Project')
st.title('PAGER-CoV-Run')
st.markdown('*Zongliang Yue, Nishant Batra, Hui-Chen Hsu, John Mountz, Jake Chen*')

st.sidebar.subheader('Data')
link = 'The COVID-19 transcriptional response data is from [GSE147507](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147507), and [GSE152418](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152418)'
st.sidebar.markdown(link, unsafe_allow_html=True)

st.sidebar.text("1.NHBE: Primary human lung epithelium.\n2.A549: Lung alveolar.\n3.Calu3:The transformed lung-derived Calu-3 cells.\n4.Lung: The lung samples.\n5.NP: The nasopharyngeal samples.\n6.PBMC: The peripheral blood mononuclear cell.")
workingdir = st.sidebar.selectbox(
    'select a cell line or tissue:',
    tuple(['NHBE','A549','Calu3','Lung','NP','PBMC']),key='workingdir'
    )

st.sidebar.markdown('You selected `%s`' % workingdir)
	
#https://github.com/streamlit/streamlit/issues/400
# get download link

#def download_link(path_temp, name_link, df=None, path_src=None):
#    path_target = Path(path_temp)
#    m = hashlib.md5()  # get unique code for this table
#    if path_temp is None or len(path_temp)==0:
#        st.markdown("**Downloadable data not available, please check temporary path symlink.**")
#        return
#    if df is not None:
#        m.update(str(len(df)).encode())
#        m.update(str(len(df["time_begin"].unique())).encode())
#        m.update(str(len(df["tag"].unique())).encode())
#    elif path_src is not None:
#        m.update(path_src.encode())
#    else:
#        st.markdown("**Eror: Neither a dataframe nor an input path were detected for downloadable data.**")
#        return
#
#    str_symlink = str(path_target.name)
#    str_unique = m.hexdigest()[:8]
#    str_url = None
#    if df is not None:
#        if st.button("Download Data", key=f"table_{str_unique}"):
#            path_write = path_target.joinpath(f"table_{str_unique}.csv")
#            if not path_write.exists():
#                df.to_csv(str(path_write), index=False)
#                str_url = f"{URL_SYMLINK_BASE}/{str_symlink}/{str(path_write.name)}"
#        else:
#            return None   # otherwise, button not clicked
#    else:       # otherwise, just make a symlink to existing path
#        path_link = path_target.joinpath(f"file_{str_unique}.csv")
#        if not path_link.exists():
#            path_link.symlink_to(path_src, True)
#        str_url = f"{URL_SYMLINK_BASE}/{str_symlink}/{str(path_link.name)}"
#    
#    st.markdown(f"[{name_link}]({str_url})")
#    return str_url

#@st.cache(allow_output_mutation=True)
def get_table_download_link(df, **kwargs):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False, sep ='\t')
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    prefix = "Download txt file for "
    if("fileName" in kwargs.keys()):
        prefix += kwargs['fileName']
    href = f'<a href="data:file/csv;base64,{b64}" download="'+kwargs['fileName']+'\.txt">'+prefix+'</a>'
    return(href)


# Return GBM treatment data as a data frame.
@st.cache(allow_output_mutation=True)
def load_treatment_data():
    #df = pd.read_csv('SampleTreatment.txt',sep="\t")
    df = pd.read_csv(workingdir+'/'+'description.txt',sep="\t")
    return(df)

# Return DEG results for JX12T pairwise comparison as data frame.
#@st.cache(allow_output_mutation=True)
def load_deg_results():
	# See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_table.html.
	expSet=[]
	names=[x for x in os.walk("./"+workingdir+"/")]
	f = []
	for (dirpath, dirnames, filenames) in os.walk("./"+workingdir+"/"):
		f.extend(filenames)
		break
	NewNames=[]
	for name in f:
		if(len(re.findall("DE_Results", name))>0):
			NewNames.append(name)
	NewNames.sort()
	for newName in NewNames:
		prefixPtn = re.compile(r'([a-z|0-9|A-Z|_]+)_sig')
		pfx=prefixPtn.findall(newName)
		tables_pd = pd.read_table("./"+workingdir+"/"+newName)
		expSet.append([pfx[0],tables_pd])
	
	return(expSet)

# Call PAGER REST API to perform hypergeometric test and return enriched PAGs associated with given list of genes as a data frame.
# See pathFun() in PAGER R SDK at https://uab.app.box.com/file/529139337869.
@st.cache(allow_output_mutation=True)
def run_pager(genes, sources, olap, sim, fdr):
	# Set up the call parameters as a dict.
	params = {}
	# Work around PAGER API form encode issue.
	if(len(genes)!=0):
		#print(genes)
		params['genes'] = '%20'.join(genes)
	else:
		params['genes'] = ''
	params['source'] = '%20'.join(sources)
	params['type'] = 'All'
	params['sim'] = sim
	params['olap'] = olap
	params['organism'] = 'All'
	params['cohesion'] = '0'
	params['pvalue'] = 0.05
	params['FDR'] = np.float64(fdr)
	params['ge'] = 1
	params['le'] = 2000

	response = requests.post('http://discovery.informatics.uab.edu/PAGER-COV/index.php/geneset/pagerapi', data=params)
#	print(response.request.body)
	return pd.DataFrame(response.json())

# gene network in PAG
@st.cache(allow_output_mutation=True)
def run_pager_int(PAGid):
	response = requests.get('http://discovery.informatics.uab.edu/PAGER-COV/index.php/pag_mol_mol_map/interactions/'+PAGid)
	return pd.DataFrame(response.json())

# generate force layout
@st.cache(allow_output_mutation=True)
def run_force_layout(G):
    pos=nx.spring_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=50, weight='weight', scale=1.0)
    return(pos)

#st.header('Query Clinical Data')
#st.markdown("These data are read from U-BRITE's *treament* programmatically from a secure call the *Unified Web Services* (UWS) API at http://ubritedvapp1.hs.uab.edu:8080/UbriteServices/getalli2b2demographics?requestorid=rdalej&cohortid=27676&format=csv.")
#clinical_data_load_state = st.text('Loading data ... ')
#clinical_data = load_clinical_data()
#clinical_data_load_state.text('Loading data ... done!')
#st.write(clinical_data)

st.header('Section 1 out of 4: Show the conditions of samples')
treatment_data = load_treatment_data()
st.table(treatment_data)
st.markdown(get_table_download_link(pd.DataFrame(treatment_data), fileName = " "+workingdir+' sample description'), unsafe_allow_html=True)  

st.header('Section 2 out of 4: Parse DEG Results')
st.markdown("These results are from a differential gene expression (DEG) analysis performed with a custom DESeq2-based pipeline on RNAseq data located in the *Omics Data Repository*.")
#st.markdown("See source code for pipeline in *Source Code Repository* at https://gitlab.rc.uab.edu/gbm-pdx/deseq2-rnaseq.")
degs = load_deg_results()

sampleNames=[]
for i in range(0,len(degs)):
    sampleNames.append(degs[i][0])

if st.checkbox('Show DEG results table', value=True):
    SampleNameButton1 = st.radio(
         "selected sample",
         sampleNames,key='DEG')
    if SampleNameButton1 in [i[0] for i in degs]:
        idx=[i[0] for i in degs].index(SampleNameButton1)
        deg=degs[idx]
        sampleName=deg[0]
        st.write('You selected: '+sampleName)
        if 'Unnamed: 0' in degs[idx][1].keys():
            degs[idx][1] = degs[idx][1].drop(['symbol'], axis=1, errors='ignore')
            degs[idx][1] = degs[idx][1].rename(columns = {"Unnamed: 0":'symbol'}) #, inplace = True
        
        st.write(degs[idx][1])

st.header('Section 3 out of 4: Run PAGER-CoV Analysis')
st.markdown("The list of significantly differentially expressed genes (DEG) is then passed to PAGER, which offers a network-accessible REST API for performing various gene-set, network, and pathway analyses.")

st.sidebar.subheader('Adjust PAGER-CoV Parameters')
link = 'The PAGER-CoV database detail [http://discovery.informatics.uab.edu/PAGER-COV/](http://discovery.informatics.uab.edu/PAGER-COV/)'
st.sidebar.markdown(link, unsafe_allow_html=True)
sources = st.sidebar.multiselect('Available Data Sources',
    ('PubChem',
    'PAGER-MSigDB','PAGER-GOA','PAGER-GOA_EXCL','PAGER-GAD','PAGER-WikiPathway','PAGER-PharmGKB',
    'PAGER-Protein Lounge','PAGER-Spike','PAGER-PheWAS','PAGER-GTEx','PAGER-Reactome','PAGER-NGS Catalog',
    'PAGER-Pfam','PAGER-GWAS Catalog','PAGER-GeneSigDB','PAGER-NCI-Nature Curated','PAGER-DSigDB',
    'PAGER-BioCarta','PAGER-KEGG',
    'Am J Respir Crit Care Med','Microbiology and Molecular Biology Reviews',
    'bioRxiv','Zenodo','Mouse Genome Informatics Database','Nature Cell Discovery',
    'GenBank (gene mapping), COVID-19 UniProt (for Geneset Description)',
    'Cell','The Annual Review of Cell and Developmental Biology','Drugbank',	
    'Cell Host and Microbe', 'Nature Medicine', 'Nature'),
    ('PubChem')
)


olap = st.sidebar.text_input("Overlap ≥", 1)
sim = st.sidebar.slider('Similarity score ≥', 0.0, 1.0, 0.1, 0.01)
fdr = st.sidebar.slider('-log2-based FDR Cutoff', 0, 300, 5, 1)
fdr = np.power(2,-np.float64(fdr))

# modified PAG enrichment
PAGERSet=pd.DataFrame()
deg_names=[]
pag_ids=[]
pags=[]
PAG_val=dict()
# Remove nan from gene list.

for deg in degs:
    deg_name=deg[0]
    deg_names.append(deg_name)
    deg_results=deg[1]
    #print(deg_results)
    genes = [x for x in deg_results['symbol'].values.tolist() if str(x) != 'nan']
    
    #pager_run_state = st.text('Calling PAGER REST API ... ')
    if len(genes) != 0:
        pager_output = run_pager(genes, sources, olap, sim, fdr)
        #pager_run_state.text('Calling PAGER REST API ... done!')
        if(st.checkbox('Show enrichment results table for sample: ' + deg_name, value=True) and len(pager_output.index)>0):#
            st.subheader('View/Filter Results')
            # Convert GS_SIZE column from object to integer dtype.
            # See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html.
            pager_output = pager_output.astype({'GS_SIZE': 'int32'})
            gs_sizes = pager_output['GS_SIZE'].tolist()
            # Figure out the min and max GS_SIZE within the PAGER output.
            min_gs_size = min(gs_sizes)
            max_gs_size = max(gs_sizes)
            # Set up a range slider. Cool!
            # See https://streamlit.io/docs/api.html#streamlit.slider.
            user_min, user_max = st.slider('GS_SIZE Range'+deg_name, max_value=max_gs_size, value=(min_gs_size, max_gs_size))
            filtered_output = pager_output[pager_output['GS_SIZE'].between(user_min, user_max)]
            st.write(filtered_output)            
            if(len(filtered_output.index)>0):
                for row in filtered_output.iloc[:,[0,1,-1]].values:
                    pag_id=str(row[0])+"_"+str(row[1])
                    pags.append(pag_id)
                    pag_ids=pag_ids+[pag_id]
                    val=-np.log(row[2])/np.log(10)
                    PAG_val[deg_name+pag_id]=val
            filtered_output['SAMPLE'] = deg_name
            PAGERSet = PAGERSet.append(filtered_output)
            st.markdown(get_table_download_link(filtered_output, fileName = deg[0]+' geneset enrichment result'), unsafe_allow_html=True)
        else:
            pager_output['SAMPLE'] = deg_name
            PAGERSet = PAGERSet.append([deg_name,pager_output])
            if(len(pager_output.index)>0):
                for row in pager_output.iloc[:,[0,1,-1]].values:
                    pag_id=str(row[0])+"_"+str(row[1])
                    pags.append(pag_id)
                    pag_ids=pag_ids+[pag_id]
                    val=-np.log(row[2])/np.log(10)
                    PAG_val[deg_name+pag_id]=val

PAGERSet = pd.DataFrame(PAGERSet)
PAGERSet['PAG_FULL'] = pag_ids
pag_ids=list(set(pag_ids))

st.write("Select the samples and narrow down the PAGs in enriched those samples")
opts = []
for deg_name in deg_names:
    opts.append((deg_name))
known_variables = {symbol: st.checkbox(f"{symbol}", value = True) for symbol in opts}
selected_pags = [key for key,val in known_variables.items() if val == True]#
pag_ids=list(set(PAGERSet[PAGERSet['SAMPLE'].isin(selected_pags)]['PAG_FULL'].tolist()))
#st.write(pag_ids)
mtx=np.zeros((len(pag_ids), len(deg_names)))
for pag_idx in range(0,len(pag_ids)):
    for name_idx in range(0,len(deg_names)):
        if(deg_names[name_idx]+pag_ids[pag_idx] in PAG_val.keys()):
            mtx[pag_idx,name_idx]=PAG_val[deg_names[name_idx]+pag_ids[pag_idx]]

            
# arbitarily order the samples
#orderExpect=['JX12T','jx14P','jx14T','x1066','x1465','x1153','x1516']
#orderExpect=['NHBE_SARS_CoV_2','NHBE_IAV','NHBE_IAVdNS1','NHBE_IFNB_4h','NHBE_IFNB_6h','NHBE_IFNB_12h',
#'A549_SARS_CoV_2','A549_HPIV3','A549_IAV','A549_RSV',
#"A549_ACE2_SARS_CoV_2","A549_ACE2_SARS_CoV_2_Rux",
#"Calu3_SARS_CoV_2"]

#st.write(treatment_data['Sample'])
orderExpect = treatment_data['Sample'].tolist()[0:]
orderIdx = [sampleNames.index(i) for i in orderExpect]
#st.write([len(pag_id) for pag_id in pag_ids])
plt = Heatmap.generateHeatmap(np.array(mtx)[::,orderIdx],np.array(deg_names)[orderIdx],pag_ids,rowCluster=True)
st.pyplot(plt)


#st.header('Section 4 out of 5: Generate the heatmap of the samples\' DEG enrichment result (' + str(len(pag_ids)) + ' PAGs)')
#from PIL import Image
#image = Image.open('./heatmap.png')
#st.image(image, caption='Sample-PAG association',
#         use_column_width=True)
#st.pyplot(caption='Sample-PAG association')

st.header('Section 4 out of 4: Generate the network of the selected PAG')
st.write('Select a PAG_ID here:')

PAGid = st.selectbox(
'Available PAG_IDs',
tuple(pag_ids)
    )

if PAGid:
    st.write("For the "+ str(PAGid)+"'s gene network")
    PAGid=re.sub("_[^_]+","",PAGid)
    geneInt=run_pager_int(PAGid)
    
    ### generate PPI data ###
    @st.cache(allow_output_mutation=True)
    def PPIgeneration(geneInt):
        idx2symbol = dict()
        idx=0
        idxPair=[]
        PPI=[]
        for pair in geneInt['data']:
            if not pair['SYM_A'] in idx2symbol.values():
                idx2symbol[idx] = pair['SYM_A']
                SYM_A_idx=idx
                idx+=1
                #print(SYM_A_idx)
            else:
                SYM_A_idx=[name for name, vals in idx2symbol.items() if vals == pair['SYM_A']][0]
            if not pair['SYM_B'] in idx2symbol.values():
                idx2symbol[idx]=pair['SYM_B']
                SYM_B_idx=idx
                idx+=1
                #print(SYM_B_idx)
            else:
                SYM_B_idx=[name for name, vals in idx2symbol.items() if vals == pair['SYM_B']][0]
            idxPair.append((SYM_A_idx,SYM_B_idx))
            PPI.append((pair['SYM_A'],pair['SYM_B']))
            
        return(idxPair,PPI,idx2symbol)

    (idxPair,PPI,idx2symbol) = PPIgeneration(geneInt)
    #st.write(PPI)
    
    # spring force layout in networkx
    import networkx as nx
    G=nx.Graph()
    G.add_nodes_from(idx2symbol.values())
    G.add_edges_from(PPI)
    pos=run_force_layout(G)
    #layout = st.sidebar.selectbox('layout',['dot',
    #                                        'neato', 
    #                                        'circo', 
    #                                        'fdp', 
    #                                        'sfdp'])
    #
    #rankdir = st.sidebar.selectbox("rankdir", ['BT', 'TB', 'LR', 'RL'])
    #ranksep = st.sidebar.slider("ranksep",min_value=0, max_value=10)
    #nodesep = st.sidebar.slider("nodesep",min_value=0, max_value=10)
    config = Config(height=500, width=700, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=False,
                  collapsible=True,              
                  node={'labelProperty':'label',"strokeColor": "black"},#, link={'labelProperty': 'label', 'renderLabel': True}
                  link={'color': "#d3d3d3"}
               )
    
    SampleNameButton = st.radio(
         "selected sample",
         sampleNames,key='network')
    colorMap = dict()
    if SampleNameButton in [i[0] for i in degs]:
        idx=[i[0] for i in degs].index(SampleNameButton)
        deg=degs[idx]
        sampleName=deg[0]
        st.write('You selected: '+sampleName)
        deg_results=deg[1]
        genesExp = [x for x in deg_results[['symbol','log2FoldChange']].values.tolist() if str(x[0]) != 'nan']

        # expression data in network
        expInNetwork=np.array(genesExp)[np.logical_or.reduce([np.array(genesExp)[:,0] == x for x in idx2symbol.values()])].tolist()
        #st.write(genesExp)
        # show expression table
        st.write("Gene expression table")
        expInNetworkArr = np.array(expInNetwork)
        expInNetworkArrSorted = np.array(sorted(expInNetworkArr,key = lambda expInNetworkArr:np.float64(expInNetworkArr[1]), reverse=True))
        DataE=pd.DataFrame(expInNetworkArrSorted)
        DataE.rename(columns={0:'symbol',1:'log2FC'},inplace=True)
        st.write(DataE)
        
        if np.size(np.array(expInNetwork))>0:
            zeroInNetwork=[[i,'0'] for i in idx2symbol.values() if i not in np.array(expInNetwork)[:,0]]
        else:
            zeroInNetwork=[[i,'0'] for i in idx2symbol.values()]
        for i in zeroInNetwork:
            expInNetwork.append(i)
            
        
        
        # And a data frame with characteristics for your nodes in networkx
        carac = pd.DataFrame({'ID':np.array(expInNetwork)[:,0], 
                              'myvalue':[np.float64(i) for i in np.array(expInNetwork)[:,1]] })
        
        # Plot it, providing a continuous color scale with cmap:
        # Here is the tricky part: I need to reorder carac, to assign the good color to each node
        carac = carac.set_index('ID')
        #carac = carac.reindex(G.nodes())
        # load network function 
        X = Network.generateNetwork(carac,pos,PPI)
        #st.pyplot(plt,caption=sampleName)
        #image = Image.open('./network.png')
        #st.image(image, caption=sampleName,
        #     use_column_width=True)
        #st.write(X.nodes)
        #st.write(X.edges)
        #st.write(carac.to_dict()["myvalue"])
        
        #st.write(newcmp)
        max_val = max([np.abs(val) for val in carac.to_dict()["myvalue"].values()])
        #st.write(max_val)
        if max_val != float(0):
            nodes = [] 
            for i in X.nodes:             
                #carac.to_dict()["myvalue"][str(i)]
                nodes.append(Node(id=i, label=str(i), size=400,  
                                  color=hex_map[int( carac.to_dict()["myvalue"][i]/max_val*colorUnit)+colorUnit])
                            ) # includes **kwargs
            edges = [Edge(source=i, label="int", target=j,color="#d3d3d3") for (i,j) in X.edges] # includes **kwargs  type="CURVE_SMOOTH"
            
            return_value = agraph(nodes=nodes, 
                          edges=edges, 
                          config=config)
            #agraph(list(idx2symbol.values()), (PPI), config)
            st.markdown(get_table_download_link(pd.DataFrame(PPI), fileName = ' '+sampleName+' '+str(PAGid)+' data for interactions'), unsafe_allow_html=True)
            st.markdown(get_table_download_link(pd.DataFrame(DataE), fileName = ' '+sampleName+' '+str(PAGid)+' data for gene expressions'), unsafe_allow_html=True)
        else:
            st.write("No expression.")
    else:
        st.write("You select nothing.")

st.header('Cite:')
st.write("PAGER-CoV analysis:")
st.write("Zongliang Yue#, Eric Zhang#, Clark Xu, Sunny Khurana, Nishant Batra, Son Dang, and Jake Y. Chen* (2021) PAGER-CoV: A Pathway, Annotated-list and Gene-signature Electronic Repository for Coronavirus Diseases Studies. Nucleic Acids Research, Volume 49, Issue D1.")
st.markdown("http://discovery.informatics.uab.edu/PAGER-COV/")
st.write("Protein-Protein Interactions (PPIs) in network construction:")
st.write("Jake Y. Chen, Ragini Pandey, and Thanh M. Nguyen, (2017) HAPPI-2: a Comprehensive and High-quality Map of Human Annotated and Predicted Protein Interactions, BMC Genomics volume 18, Article number: 182")
st.markdown("http://discovery.informatics.uab.edu/HAPPI/")        
        
##for idx in range(0,len(degs)):
##    deg=degs[idx]
##    sampleName=deg[0]
##    deg_results=deg[1]
##    genesExp = [x for x in deg_results[['symbol','log2FoldChange']].values.tolist() if str(x[0]) != 'nan']
##    # expression data in network
##    expInNetwork=np.array(genesExp)[np.logical_or.reduce([np.array(genesExp)[:,0] == x for x in idx2symbol.values()])].tolist()
##    if np.size(np.array(expInNetwork))>0:
##        zeroInNetwork=[[i,'0'] for i in idx2symbol.values() if i not in np.array(expInNetwork)[:,0]]
##    else:
##        zeroInNetwork=[[i,'0'] for i in idx2symbol.values()]
##    for i in zeroInNetwork:
##        expInNetwork.append(i)
##    # And a data frame with characteristics for your nodes
##    carac = pd.DataFrame({ 'ID':np.array(expInNetwork)[:,0], 'myvalue':[np.float(i) for i in np.array(expInNetwork)[:,1]] })
##    # Plot it, providing a continuous color scale with cmap:
##    # Here is the tricky part: I need to reorder carac, to assign the good color to each node
##    carac= carac.set_index('ID')
##    carac=carac.reindex(G.nodes())
##
##    plt=Network.generateNetwork(carac,pos,PPI)
##    st.pyplot(caption=sampleName)

##plt.savefig("./network.png", dpi=100,transparent = True, bbox_inches = 'tight', pad_inches = 0)
##
##from PIL import Image
##image = Image.open('./network.png')
##st.image(image, #caption='Sample-PAG association',
##         use_column_width=True)