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
import scipy.spatial as sp, scipy.cluster.hierarchy as hc

#http://seaborn.pydata.org/generated/seaborn.clustermap.html
import seaborn as sns; sns.set(color_codes=True)
from matplotlib import rcParams

def generateheatmap(mtx,deg_names,pag_ids,**kwargs):
    plt.figure(figsize=(5,5))
    # parameters in the heatmap setting 
    width_ratio = 1
    annotationSize = 6
    font_size = 12
    rowCluster = False
    colCluster = False
    
    if 'width_ratio' in kwargs.keys():
        width_ratio = kwargs['width_ratio']      
    if 'annotationSize' in kwargs.keys():
        annotationSize = kwargs['annotationSize']
    if 'rowCluster' in kwargs.keys():
        rowCluster = kwargs['rowCluster']            
    if 'colCluster' in kwargs.keys():
        colCluster = kwargs['colCluster']
        
    outputdir = kwargs['outputdir'] if 'outputdir' in kwargs.keys() else ""
    if deg_names.size > 1:
        #fig, ax = plt.subplots(figsize=(5/len(pag_ids), length))
        # {‘ward’, ‘complete’, ‘average’, ‘single’}
        col_linkage = hc.linkage(sp.distance.pdist(mtx.T), method='average')
        row_linkage = hc.linkage(sp.distance.pdist(mtx), method='average')
    
    # load the color scale using the cm
    #top = cm.get_cmap('Blues_r', 56)
    bottom = cm.get_cmap('Reds', 56)
    newcolors = np.vstack(
                            (
                                #top(np.linspace(0, 1, 56)),
                                ([[0,0,0,0.1]]),
                                bottom(np.linspace(0, 1, 56))
                            )
                         )
    newcmp = ListedColormap(newcolors, name='RedBlue')
    # set the balance point of the expression to 0
    f_max = np.max(mtx)
    f_min = np.min(mtx)
    
    if(abs(f_max)>abs(f_min)):
        Bound=abs(f_max)
    else:
        Bound=abs(f_min)
       
    # figure size in inches
    #rcParams['figure.figsize'] = 3,8.27
    expMtxsDF = pd.DataFrame(mtx)
    expMtxsDF.columns = deg_names
    expMtxsDF.index = pag_ids
    #sns.set(font_scale=1,rc={'figure.figsize':(3,20)})
    
    
    #print(rowCluster == True and int(deg_names.size) > 1)

    #print(int(deg_names.size))
    if(rowCluster == True and colCluster == True and int(deg_names.size) > 1): 
        g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,col_linkage=col_linkage,row_linkage=row_linkage,  yticklabels=True,
                      annot=True,annot_kws={"size": annotationSize})        
    
    elif(rowCluster == True and int(deg_names.size) > 1): 
        g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,col_cluster=False,row_linkage=row_linkage, yticklabels=True,
                      annot=True,annot_kws={"size": annotationSize})
    elif(colCluster == True and int(deg_names.size) > 1): 
        g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,col_linkage=col_linkage,row_cluster=False,  yticklabels=True,
                      annot=True,annot_kws={"size": annotationSize})  
    else:
        if int(deg_names.size) == 1:
            expMtxsDF = expMtxsDF.sort_values(by=list(deg_names), ascending=False)
        g = sns.clustermap(expMtxsDF,cmap=newcmp,vmax=Bound,vmin=0,row_cluster=False,col_cluster=False,  yticklabels=True,
                      annot=True,annot_kws={"size": annotationSize})  
        #g.ax_row_dendrogram.set_xlim([0,0]) 
    #plt.subplots_adjust(top=0.9) # make room to fit the colorbar into the figure
    ### rotation of labels of x-axis and y-axis
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize= font_size)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize= font_size-2)
    hm = g.ax_heatmap.get_position()
    scale_factor = len(pag_ids)/40
    if scale_factor <  0.5:
        scale_factor = 0.5
    #max_content_length = (40/max([len(pag) for pag in pag_ids]))
    #if max_content_length >10:
    #    max_content_length = 10
    #width_ratio = width_ratio * max_content_length * int(deg_names.size**2)
    #if scale_factor<3 or scale_factor>7:
    #    width_ratio = width_ratio *1.5
    # to change the legends location
    g.ax_heatmap.set_position([hm.x0*scale_factor, hm.y0*scale_factor, hm.width*width_ratio*scale_factor, hm.height*scale_factor])
    col = g.ax_col_dendrogram.get_position()
    g.ax_col_dendrogram.set_position([col.x0*scale_factor, col.y0*scale_factor, col.width*width_ratio*scale_factor, col.height*0.5]) #
    row = g.ax_row_dendrogram.get_position()
    g.ax_row_dendrogram.set_position([row.x0*scale_factor, row.y0*scale_factor, row.width*scale_factor, row.height*scale_factor]) #
    #for i, ax in enumerate(g.fig.axes):   ## getting all axes of the fig object
    #    ax.set_xticklabels(ax.get_xticklabels(), rotation = rotation)
    ### color bar position and title ref: https://stackoverflow.com/questions/67909597/seaborn-clustermap-colorbar-adjustment
    ### color bar position adjustment
    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([x0, _y0*scale_factor+0.1, row.width*scale_factor, 0.05])
    g.ax_cbar.set_title('-log2 FDR')        
    bottom, top = g.ax_heatmap.get_ylim()
    plt.rcParams["axes.grid"] = False       
    return(plt)


# color in hex_map format
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
st.header('An online interactive analytical platform for COVID-19 functional genomic downstream analysis')
st.markdown('*Zongliang Yue, Nishant Batra, Hui-Chen Hsu, John Mountz, and Jake Chen*')

st.sidebar.subheader('Data')
link = 'The COVID-19 transcriptional response data is from [GEO database](https://www.ncbi.nlm.nih.gov/geo/)'
st.sidebar.markdown(link, unsafe_allow_html=True)

st.sidebar.text("1.NHBE: Primary human lung epithelium.\n2.A549: Lung alveolar.\n3.Calu3:The transformed lung-derived Calu-3 cells.\n4.Lung: The lung samples.\n5.NP: The nasopharyngeal samples.\n6.PBMC: The peripheral blood mononuclear cell.\n7.Leukocyte: The leukocytes.\n8.hiPSC:Human induced pluripotent stem cell-derived cardiomyocytes\n9.Liver Organoid.\n10.Pancreas Organoid")
workingdir = st.sidebar.selectbox(
    'select a cell line or tissue:',
    tuple(['NHBE','A549','Calu3','Lung','NP','PBMC','Leukoycyte','hiPSC','Liver Organoid','Pancreas Organoid']),key='workingdir'
    )

st.sidebar.markdown('You selected `%s`' % workingdir)
	
#https://github.com/streamlit/streamlit/issues/400
# get download link

#@st.cache(allow_output_mutation=True)
def get_table_download_link(df, **kwargs):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True, sep ='\t')
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

# pag_ranked_gene in PAG
#@st.cache(allow_output_mutation=True)
def pag_ranked_gene(PAGid):
	response = requests.get('http://discovery.informatics.uab.edu/PAGER-COV/index.php/genesinPAG/viewgenes/'+PAGid)
	return pd.DataFrame(response.json()['gene'])

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

treatment_data = load_treatment_data()
if 'Sample' in treatment_data.columns:
    treatment_data.set_index('Sample', inplace=True)
degs = load_deg_results()

sampleNames=[]
for i in range(0,len(degs)):
    sampleNames.append(degs[i][0])
    
orderExpect = treatment_data.index.tolist()[0:]
orderIdx = [sampleNames.index(i) for i in orderExpect]
    

st.header('Section 1 out of 4: Show the conditions of samples')
st.table(treatment_data)
st.markdown(get_table_download_link(treatment_data, fileName = " "+workingdir+' sample description'), unsafe_allow_html=True)  

st.header('Section 2 out of 4: Parse DEG Results')
st.markdown("These results are from a differential gene expression (DEG) analysis performed with a custom DESeq2-based pipeline on RNAseq data located in the *Omics Data Repository*.")
#st.markdown("See source code for pipeline in *Source Code Repository* at https://gitlab.rc.uab.edu/gbm-pdx/deseq2-rnaseq.")

if st.checkbox('Show DEG results table', value=True):
    SampleNameButton1 = st.radio(
         "selected sample",
         [sampleNames[order] for order in orderIdx],key='DEG')
    if SampleNameButton1 in [i[0] for i in degs]:
        idx=[i[0] for i in degs].index(SampleNameButton1)
        deg=degs[idx]
        sampleName=deg[0]
        st.write('You selected: '+sampleName)
        if 'Unnamed: 0' in degs[idx][1].keys():
            degs[idx][1] = degs[idx][1].drop(['symbol'], axis=1, errors='ignore')
            degs[idx][1] = degs[idx][1].rename(columns = {"Unnamed: 0":'symbol'}) #, inplace = True
        
        st.write(degs[idx][1])
        st.markdown(get_table_download_link(pd.DataFrame(degs[idx][1]), fileName = degs[idx][0]+' DEG list result'), unsafe_allow_html=True)
st.header('Section 3 out of 4: Run PAGER-CoV Analysis')
st.markdown("The list of significantly differentially expressed genes (DEG) is then passed to Pathways, Annotated gene lists, and Gene signatures Electronic Repository (PAGER), which offers a network-accessible REST API for performing various gene-set, network, and pathway analyses.")

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
sim = st.sidebar.slider('Similarity score ≥', 0.0, 1.0, 0.12, 0.01)
fdr = st.sidebar.slider('-log2-based FDR Cutoff', 0, 300, 5, 1)
fdr = np.power(2,-np.float64(fdr))

# modified PAG enrichment
PAGERSet=pd.DataFrame()
deg_names=[]
pag_ids=[]
pags=[]
PAG_val=dict()
# Remove nan from gene list.
for idx in orderIdx:
    deg = degs[idx]
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
            if(len(pager_output.index)>0):
                pager_output['SAMPLE'] = deg_name
                PAGERSet = PAGERSet.append([deg_name,pager_output])
                for row in pager_output.iloc[:,[0,1,-1]].values:
                    pag_id=str(row[0])+"_"+str(row[1])
                    pags.append(pag_id)
                    pag_ids=pag_ids+[pag_id]
                    val=-np.log(row[2])/np.log(10)
                    PAG_val[deg_name+pag_id]=val

PAGERSet = pd.DataFrame(PAGERSet)
#st.write(PAGERSet.shape[1])
if PAGERSet.shape[1] < 2:
    st.write("No enriched PAGs found. Try a lower similarity score or a lower -log2-based FDR cutoff and rerun.")
    st.stop()
#st.write(PAGERSet)
#st.write(pag_ids)
PAGERSet['PAG_FULL'] = pag_ids
pag_ids=list(set(pag_ids))


st.write("Select the samples and narrow down the PAGs in enriched those selected samples")
opts = []
for deg_name in deg_names:
    opts.append((deg_name))
known_variables = {symbol: st.checkbox(f"{symbol}", value = True) for symbol in opts}
selected_pags = [key for key,val in known_variables.items() if val == True]#
if(len(selected_pags) == 0):
    st.write("Please select at least one sample to generate the heatmap.")
    st.stop()   
pag_ids=list(set(PAGERSet[PAGERSet['SAMPLE'].isin(selected_pags)]['PAG_FULL'].tolist()))
#st.write(pag_ids)
mtx=np.zeros((len(pag_ids), len(deg_names)))
for pag_idx in range(0,len(pag_ids)):
    for name_idx in range(0,len(deg_names)):
        if(deg_names[name_idx]+pag_ids[pag_idx] in PAG_val.keys()):
            mtx[pag_idx,name_idx]=PAG_val[deg_names[name_idx]+pag_ids[pag_idx]]



#st.write([len(pag_id) for pag_id in pag_ids])

width_ratio_heatmap = st.slider('Width ratio of heatmap (increase to widen the heatmap)', 0.1, 5.0, 1.0, 0.1)

### heatmap ###
heatmapBtn = st.button("Generate the heatmap")
if heatmapBtn == True:
    plt = generateheatmap(np.array(mtx)[::,orderIdx]
                              ,np.array(deg_names)[orderIdx]
                              ,pag_ids
                              ,rowCluster=True
                              ,colCluster = False
                              ,width_ratio=width_ratio_heatmap)
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
    ID_only = re.sub("([A-Z0-9]+)_[^_]*","\\1",str(PAGid))
    link = "For the selected PAG "+ str(PAGid)+"'s gene information. (http://discovery.informatics.uab.edu/PAGER-COV/index.php/geneset/view/"+ID_only+")"
    st.markdown(link, unsafe_allow_html=True)
    PAGid=re.sub("_[^_]+","",PAGid)
    geneInt=run_pager_int(PAGid)
    geneRanked=pag_ranked_gene(PAGid)
    #st.write(geneRanked)
    idx2symbol = dict()
    symbol2idx = dict()
    symbol2size = dict()
    idx=0
    geneRanked['RP_SCORE'].fillna(0.1, inplace=True)
    geneRanked['RP_SCORE'] = geneRanked['RP_SCORE'].astype(float)
    geneRanked['node_size'] = geneRanked['RP_SCORE'] *4
    st.write(geneRanked)
    st.markdown(get_table_download_link(geneRanked, fileName = " "+workingdir+" "+str(PAGid)), unsafe_allow_html=True) 
    
    for gene_idx in range(0,geneRanked.shape[0]):

        gene = geneRanked.iloc[gene_idx,]
        #st.write(gene)
        symbol2idx[gene['GENE_SYM']] = str(idx)
        #st.write(gene['RP_SCORE'])
        #symbol2size[gene['GENE_SYM']] = gene['RP_SCORE']
        if(gene['RP_SCORE'] is not None):
            symbol2size[gene['GENE_SYM']] = gene['node_size']
        else:
            symbol2size[gene['GENE_SYM']] = 1
        idx2symbol[str(idx)] = gene['GENE_SYM']
        idx+=1
    ### generate PPI data ###
    @st.cache(allow_output_mutation=True)
    def PPIgeneration(geneInt,symbol2idx):      
        idxPair=[]
        PPI=[]
        for pair in geneInt['data']:
            idxPair.append((symbol2idx[pair['SYM_A']],symbol2idx[pair['SYM_B']]))
            PPI.append((pair['SYM_A'],pair['SYM_B']))
            
        return(idxPair,PPI,idx2symbol)

    (idxPair,PPI,idx2symbol) = PPIgeneration(geneInt,symbol2idx)
    #st.write(PPI)
    
    # spring force layout in networkx
    import networkx as nx
    G=nx.Graph()
    G.add_nodes_from(idx2symbol.values())
    G.add_edges_from(PPI)
    pos=run_force_layout(G)


    
    #SampleNameButton = st.radio(
    #     "selected sample",
    #     sampleNames,key='network')
    colorMap = dict()
    
    #if SampleNameButton in [i[0] for i in degs]:    
        #idx=[i[0] for i in degs].index(SampleNameButton)
    for idx in orderIdx:        
        deg=degs[idx]
        sampleName=deg[0]
        config = Config(height=500, width=700, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=False,
              collapsible=True,              
              node={'labelProperty':'label',"strokeColor": "black"},
              #, link={'labelProperty': 'label', 'renderLabel': True}
              link={'color': "#d3d3d3"},
              key="agraph_"+sampleName
           )
        st.write("Sample:"+sampleName)
        deg_results=deg[1]
        genesExp = [x for x in deg_results[['symbol','log2FoldChange','lfcSE']].values.tolist() if str(x[0]) != 'nan']

        # expression data in network
        expInNetwork=np.array(genesExp)[np.logical_or.reduce([np.array(genesExp)[:,0] == x for x in idx2symbol.values()])].tolist()
        #st.write(genesExp)
        # show expression table
        st.write("Gene expression within the selected PAG:")
        expInNetworkArr = np.array(expInNetwork)
        expInNetworkArrSorted = np.array(sorted(expInNetworkArr,key = lambda expInNetworkArr:np.float64(expInNetworkArr[1]), reverse=True))
        DataE=pd.DataFrame(expInNetworkArrSorted)
        DataE.rename(columns={0:'symbol',1:'log2FC',2:'S.E.'},inplace=True)
        st.write(DataE)
        
        if np.size(np.array(expInNetwork))>0:
            zeroInNetwork=[[i,'0','0'] for i in idx2symbol.values() if i not in np.array(expInNetwork)[:,0]]
        else:
            zeroInNetwork=[[i,'0','0'] for i in idx2symbol.values()]
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
                nodes.append(Node(id=i, label=str(i), size=symbol2size[str(i)],#400,  
                                  color=hex_map[int( carac.to_dict()["myvalue"][i]/max_val*colorUnit)+colorUnit])
                            ) # includes **kwargs
            edges = [Edge(source=i, label="int", target=j,color="#d3d3d3") for (i,j) in X.edges] # includes **kwargs  type="CURVE_SMOOTH"
            
            return_value = agraph(
                nodes=nodes, 
                edges=edges, 
                config=config)
            #agraph(list(idx2symbol.values()), (PPI), config)
            st.markdown(get_table_download_link(pd.DataFrame(PPI), fileName = ' '+sampleName+' '+str(PAGid)+' data for interactions'), unsafe_allow_html=True)
            st.markdown(get_table_download_link(pd.DataFrame(DataE), fileName = ' '+sampleName+' '+str(PAGid)+' data for gene expressions'), unsafe_allow_html=True)
        else:
            st.write("No expression.")
    #else:
    #    st.write("You select nothing.")

st.header('Cite:')
st.write("PAGER-CoV analysis:")
st.write("Zongliang Yue#, Eric Zhang#, Clark Xu, Sunny Khurana, Nishant Batra, Son Dang, and Jake Y. Chen* (2021) PAGER-CoV: A Pathway, Annotated-list and Gene-signature Electronic Repository for Coronavirus Diseases Studies. Nucleic Acids Research, Volume 49, Issue D1.")
st.markdown("http://discovery.informatics.uab.edu/PAGER-COV/")
st.write("Protein-Protein Interactions (PPIs) in network construction:")
st.write("Jake Y. Chen, Ragini Pandey, and Thanh M. Nguyen, (2017) HAPPI-2: a Comprehensive and High-quality Map of Human Annotated and Predicted Protein Interactions, BMC Genomics volume 18, Article number: 182")
st.markdown("http://discovery.informatics.uab.edu/HAPPI/")        

st.header('About us:')
st.write(f"If you have questions or comments about the database contents, please email Dr. Jake Chen, jakechen@uab.edu.")
st.write("If you need any technical support, please email Zongliang Yue, zongyue@uab.edu.")
st.write("Our lab: AI.MED Laboratory, University of Alabama at Birmingham, Alabama, USA. Link: http://bio.informatics.uab.edu/")
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