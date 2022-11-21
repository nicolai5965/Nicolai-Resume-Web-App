###------------------------------------------------------------------------------------------------------------###
#Importing libraries
from pathlib import Path
import os
import streamlit as st
from PIL import Image
import webbrowser
import numpy as np 
import pandas as pd
import math
from numba import njit , jit
from tqdm import trange, tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from collections import Counter
from stqdm import stqdm
import matplotlib.pyplot as plt
os.environ['NUMEXPR_MAX_THREADS'] = '12'
###------------------------------------------------------------------------------------------------------------###
# --- GENERAL SETTINGS ---
resume_file = "pages/CV.pdf"
Bachelorproject_file = "pages/Bachelorproject.pdf"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/nicolai-s%C3%B8derberg-907680238/",
    "GitHub": "https://github.com/nicolai5965",
}
###------------------------------------------------------------------------------------------------------------###
### Sidebar
st.sidebar.write('\n')

LinkedIn_link = '[My LinkedIn](https://www.linkedin.com/in/nicolai-s%C3%B8derberg-907680238/)'
st.sidebar.markdown(LinkedIn_link, unsafe_allow_html=True)

GitHub_link = '[My GitHub repo](https://github.com/nicolai5965)'
st.sidebar.markdown(GitHub_link, unsafe_allow_html=True)

with open(resume_file, "rb") as pdf_file:
    PDFbyte_CV = pdf_file.read()
    
st.sidebar.download_button(
    label = "Download Resume 👈",
    data = PDFbyte_CV,
    file_name = "Resume/CV.pdf",
    mime ="application/octet-stream",)

st.sidebar.write("---")

# if st.sidebar.button('Kaggle: TrackML Challenge'):
#     webbrowser.open_new_tab("https://www.kaggle.com/competitions/trackml-particle-identification/overview")
# if st.sidebar.button("Bachelor's project: Old code"):
#     webbrowser.open_new_tab("https://github.com/nicolai5965/Bachelor-project-Nicolai-og-Simon-")

# if st.sidebar.button("Bachelor's project: Remade"):
#     webbrowser.open_new_tab("https://github.com/nicolai5965/Bachelor-project-remastered")

Kaggle_link = "[Kaggle: TrackML Challenge](https://www.kaggle.com/competitions/trackml-particle-identification/overview)"
st.sidebar.markdown(Kaggle_link, unsafe_allow_html=True)

GitHub_link_old_code = '[Bachelor project: Old code](https://github.com/nicolai5965)'
st.sidebar.markdown(GitHub_link_old_code, unsafe_allow_html=True)  

GitHub_link_new_code = '[Bachelor project: Remade](https://github.com/nicolai5965)'
st.sidebar.markdown(GitHub_link_new_code, unsafe_allow_html=True)  
  
with open(Bachelorproject_file, "rb") as pdf_file:
    PDFbyte_project = pdf_file.read()

st.sidebar.download_button(
    label = "Download Bachelor's project",
    data = PDFbyte_project,
    file_name = Bachelorproject_file,
    mime = "application/octet-stream",)
st.sidebar.write("---")
###------------------------------------------------------------------------------------------------------------###
## Instroduction 
st.header("Introduction:")
Introduction = """This project was my physics bachelor's project that I did with my classmate Simon. We got the idea from our supervisor Stefania. We could either do a convolutional neural network, which had been done multiple times before, or we could do a graph neural network which isn't as widespread as CNN. We chose the GNN because it sounded super interesting, especially since it isn't as widespread. The reason I'm making this page is that I have made some significant changes to the parts of the code. These changes makes the code run significantly faster. With some parts taking multipul hours down to a couple of minutes. 
If you want to read our completed report you can download it from the sidebar, or look at the chanllenge you self click the Kaggle link.
"""
st.write(Introduction)

st.write('\n')
st.write("---")
###------------------------------------------------------------------------------------------------------------###
# Track image:
st.header("Here you can see an image of a end prediction of the track of particles, from our GNN")
front_image = Image.open("pages/Images/BP front pic.PNG")
st.image(front_image,
        width=500,
        output_format="PNG")

###----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
page_info = st.radio("Select date format",
            ('Read description:', 'Test GNN:', 'View detector setup'),
            label_visibility="hidden" )

if page_info == 'Read description:':
    ###------------------------------------------------------------------------------------------------------------###
    # Abstract:
    Abstract = """Due to the significant scale of data produced in High Luminosity LHC, traditional particle tracking methods will require more time reconstructing particle tracks. To confront
    this problem, various different machine learning techniques have been tested. A method
    that has produced promising results is a Graph Neural Network, which can predict possible
    track candidates through the coordinates of different hits in the detector. In this investigation, we apply the use of a neural network to classify potential track segments and from
    this aim to reconstruct the trajectories of individual particles. This network is trained and
    tested using the public data-sets provided from the TrackML Particle Tracking Challenge.
    The aim of this investigation is to analyse how fast this kind of neural network can predict
    possible track segments as well as how accurately it can predict them. Finally we aim to
    reconstruct the actual trajectories of individual particles using the predicted track segments
    from the neural network.
    """
    st.write('\n')
    st.write("---")
    st.header("Abstract:")
    st.write(Abstract)
    ###------------------------------------------------------------------------------------------------------------###
    # Getting started:
    Getting_started = """We have only looked at Kaggles 100 events file. This is because we didn't have the time to run all 8850 events when 100 took over an hour with the old code. 
    When getting the 100 events file there are four file types Hits, Truth, Particles and Cells. 
    We found no use in the cells file, so I have made some code that can remove all the Cell files from the 100 events folder. This can be found under the Ramade github page.
    From here we can be getting starting with reading all the events that we wants to look at. 
    """
    st.write('\n')
    st.write("---")
    st.header("Getting started:")
    st.write(Getting_started)
    ###------------------------------------------------------------------------------------------------------------###
    # Cleaning data
    Cleaning_data = """There are some of the particles/hits that has to be removed. Why this has to happen can be read in the finished project.
    """
    st.write('\n')
    st.write("---")
    st.header("Cleaning data:")
    st.write(Cleaning_data)
    ###------------------------------------------------------------------------------------------------------------###
    # finding_nodes
    finding_nodes = """We decided to make a GNN that could take in the features of the different nodes, and then predict whichs nodes where connected here by findes the right edges.
    """
    st.write('\n')
    st.write("---")
    st.header("Finding the node features:")
    st.write(finding_nodes)
    ###------------------------------------------------------------------------------------------------------------###
    # using MLP
    MLP_C = """With the limited time that we had, we chose to go along and used Sklearns MLP classifier as the base of out GNN."""
    st.write('\n')
    st.write("---")
    st.header("Sklearns MLP classifier")
    st.write(MLP_C)
    ###------------------------------------------------------------------------------------------------------------###
    st.write('\n')
    st.write("---")
    st.header("All of above and more can be read in our bachelor's project")

if page_info == "Test GNN:":
# ###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
#     st.header("Data selection parameters:")
#     mlp_settings = '<p style="font-family:sans-serif; color:White; font-size: 22px;"> All settings are on a default that I found give the best results without taking hours to run </p>'
#     st.markdown(mlp_settings, unsafe_allow_html=True)
#     ###------------------------------------------------------------------------------------------------------------###
#     st.write("Do you want to set data selection parameters yourself?")
#     col1, col2 , col3, col4, col5, col6 = st.columns(6)
#     with col1:
#         data_para_set = st.select_slider("data_para_set", ["No", "Yes"], label_visibility = "hidden")
    
#     if data_para_set == "No":
#         event_nr = 10
#         detector_sections = [8, 13, 17]
#         pt_values = (1.0, 1.25)
   
#     if data_para_set == "Yes":
#         st.header('Select number of events:')
#         col1, col2 , col3 = st.columns(3)
#         with col1:
#             event_nr = st.number_input("event_nr", min_value = 2, max_value = 100, value = 10, label_visibility = "hidden")
#             #st.write('The event amount = ', number)

#         ###------------------------------------------------------------------------------------------------------------###
#         st.header("Selecet which sections/volume_id you would like to look at:")
#         detector_sections = st.multiselect(
#             "Select Countrys:",
#             options = [7, 8, 12, 13, 14, 16, 17, 18],
#             default = [8, 13, 17],
#             label_visibility = "hidden",)

#         st.header("Selecet transverse momentum range:")
#         pt_values = st.slider(
#         'The bigger the range the longer it takes the code to run:',
#         0.0, 3.0, (1.0, 1.25))
#         ###------------------------------------------------------------------------------------------------------------###
#         if st.button('View detector sections:'):
#             logo_image = Image.open("pages/Images/Detector volume id.png")
#             st.image(logo_image,
#                     width=650,
#                     output_format="PNG")
    
#     st.write("The current data selection parameters are set to:")
#     st.write(f"""Number of events:, {event_nr}. Detector sections: {detector_sections}. Pt value range: {pt_values}.""")
    
#     # run_data_reading = '<p style="font-family:sans-serif; color:White; font-size: 22px;"> Do you want to read, clean and find nodes? </p>'
#     # st.markdown(run_data_reading, unsafe_allow_html=True)
#     ###-----------------------------------------------------------------------------------------------------------------------------------------------------###
#     st.header("MLP parameters:")
#     mlp_settings = '<p style="font-family:sans-serif; color:White; font-size: 22px;"> All settings are on a default that I found give the best results without taking hours to run </p>'
#     st.markdown(mlp_settings, unsafe_allow_html=True)
    
###-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
    st.write("Do you want to set MLP parameters yourself?")
    col1, col2 , col3, col4, col5, col6 = st.columns(6)
    with col1:
        mlp_para_set = st.select_slider("", ["No", "Yes"])
    
    if mlp_para_set == "No":
        hidden_layers = (100, 200, 100)
        max_iter = 100
        activation = "relu"
        solver = "adam"
        learning_rate = "constant"
        
    if mlp_para_set == "Yes":
        st.write("Selecet number hidden layers:")
        st.write("Min: 2", "Max: 4")
        col1, col2 , col3 = st.columns(3)
        with col1:
            nr_hidden_layers = st.number_input("col1_number1", min_value = 2, max_value = 4, value = 3, label_visibility = "hidden")

#         if nr_hidden_layers == 2:
#             st.write("Selecet hidden layers:")
#             col1, col2 , col3 = st.columns(3)
#             with col1:
#                 col2_number1 = st.number_input("col2_number1", min_value = 1, max_value = 300, value = 100, label_visibility = "hidden")

#             with col2:
#                 col2_number2 = st.number_input("col2_number2", min_value = 1, max_value = 300, value = 200, label_visibility = "hidden")

#             hidden_layers = (col2_number1, col2_number2)

#         if nr_hidden_layers == 3:
#             st.write("Selecet hidden layers:")
#             col1, col2 , col3 = st.columns(3)
#             with col1:
#                 col3_number1 = st.number_input("col3_number1", min_value = 1, max_value = 300, value = 100, label_visibility = "hidden")

#             with col2:
#                 col3_number2 = st.number_input("col3_number2", min_value = 1, max_value = 300, value = 200, label_visibility = "hidden")

#             with col3:
#                 col3_number3 = st.number_input("col3_number3", min_value = 1, max_value = 300, value = 100, label_visibility = "hidden")

#             hidden_layers = (col3_number1, col3_number2, col3_number3)

#         if nr_hidden_layers == 4:
#             st.write("Selecet hidden layers:")
#             col1, col2 , col3, col4 = st.columns(4)
#             with col1:
#                 col4_number1 = st.number_input("col4_number1", min_value = 1, max_value = 300, value = 100, label_visibility = "hidden")

#             with col2:
#                 col4_number2 = st.number_input("col4_number2", min_value = 1, max_value = 300, value = 200, label_visibility = "hidden")

#             with col3:
#                 col4_number3 = st.number_input("col4_number3", min_value = 1, max_value = 300, value = 100, label_visibility = "hidden")

#             with col4:
#                 col4_number4 = st.number_input("col4_number4", min_value = 1, max_value = 300, value = 50, label_visibility = "hidden")   

#             hidden_layers = (col4_number1, col4_number2, col4_number3, col4_number4)  
        ###------------------------------------------------------------------------------------------------------------###
        st.write("Selecet maximum number of iterations:")
        col1, col2 , col3 = st.columns(3)
        with col1:
            max_iter = st.number_input("max_iter", min_value = 1, max_value = 400, value = 100, label_visibility = "hidden")
        ###------------------------------------------------------------------------------------------------------------###
        col1, col2 , col3 = st.columns(3)
        with col1:
            activation = st.radio("Select MLP activation:",
                        ("relu", "identity", "logistic", "tanh"))
        ###------------------------------------------------------------------------------------------------------------###
        with col2:
            solver = st.radio("Select MLP solver:",
                        ("adam", "sgd", "lbfgs"))
        ###------------------------------------------------------------------------------------------------------------###
        with col3:
            learning_rate = st.radio("Select MLP learning_rate:",
                        ("constant", "invscaling", "adaptive"))
        
    st.write("The MLP parameters are set to:")
    st.write(f"hidden_layers:", {hidden_layers}, " ,", "max_iter:", {max_iter}, " ,", "solver:", {solver}, " ,", "activation:", {activation}, " ,", "learning_rate:", {learning_rate})

###---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###
#     run_MLP_test = '<p style="font-family:sans-serif; color:White; font-size: 22px;"> Do you want to read, clean, find nodes and run the MLP? </p>'
#     st.markdown(run_MLP_test, unsafe_allow_html=True)    
#     col1, col2 , col3, col4, col5, col6 = st.columns(6)
#     with col1:
#         run_MLP = st.select_slider("run_mlp", ["No", "Yes"], label_visibility = "hidden")
    
#     if run_MLP == "No":
#         st.write("Waiting to read, clean, find nodes and run the MLP")
    
#     if run_MLP == "Yes":    
#         def reading_data(data_path, events_called):
#             data_merge = []
#             particles = []
#             for i in stqdm(range(0,events_called)):
#                 if i < 10:
#                     hist_data = pd.read_csv(f'{data_path}/event00000100%d-hits.csv' % i)
#                     particles_data = pd.read_csv(f'{data_path}/event00000100%d-particles.csv'% i)
#                     truth_data = pd.read_csv(f'{data_path}/event00000100%d-truth.csv'% i)
#                     particles.append(particles_data)
#                     data_merge.append(pd.merge(hist_data, truth_data, on='hit_id', how='outer'))
#                 if i >= 10:
#                     hist_data = pd.read_csv(f'{data_path}/event0000010%d-hits.csv' % i)
#                     particles_data = pd.read_csv(f'{data_path}/event0000010%d-particles.csv'% i)
#                     truth_data = pd.read_csv(f'{data_path}/event0000010%d-truth.csv'% i)
#                     particles.append(particles_data)
#                     data_merge.append(pd.merge(hist_data, truth_data, on='hit_id', how='outer'))
#             return data_merge , particles

        
#         def remove_volume_ids(data_H_T, volumn_list):
#             data_H_T = data_H_T.loc[data_H_T['volume_id'].isin(volumn_list)]
#             return data_H_T


#         def remove_weight_0(data_H_T):
#             data_H_T = data_H_T.loc[data_H_T["weight"] > 0]
#             return data_H_T


#         def remove_particle_id_0(data_H_T):
#             data_H_T = data_H_T.loc[data_H_T["particle_id"] > 0]
#             return data_H_T


#         def remove_nhits_less_3(data_P):
#             data_P = data_P.loc[data_P["nhits"] > 3]
#             return data_P


#         def same_layer_filter(data_H_T):
#             data_H_T = data_H_T.drop_duplicates(subset = ["particle_id", "volume_id", "layer_id"])
#             small_occurrence = data_H_T["particle_id"].value_counts() > 3
#             data_H_T = data_H_T[data_H_T["particle_id"].isin(small_occurrence[small_occurrence].index)]
#             return data_H_T


#         def pt_cutter(data_P, pt_cut_start, pt_cut_end):
#             data_P["Pt"] = (data_P.px**2+data_P.py**2)**(1/2)
#             data_P = data_P.loc[(data_P["Pt"] >= pt_cut_start) & (data_P["Pt"] <= pt_cut_end)]
#             return data_P


#         def full_data_clean(data_H_T, data_P, volumn_list, pt_cut_start, pt_cut_end):
#             data_H_T = remove_volume_ids(data_H_T, volumn_list)
#             data_H_T = remove_weight_0(data_H_T)
#             data_H_T = remove_particle_id_0(data_H_T)
#             data_H_T = same_layer_filter(data_H_T)

#             data_P = pt_cutter(data_P, pt_cut_start, pt_cut_end)
#             data_P = remove_nhits_less_3(data_P)

#             particle_id_list = list(set(data_H_T.particle_id) - set(data_P.particle_id))
#             data_H_T = data_H_T[~data_H_T['particle_id'].isin(particle_id_list)]

#             particle_id_list = list(set(data_P.particle_id) - set(data_H_T.particle_id))
#             data_P = data_P[~data_P['particle_id'].isin(particle_id_list)]
#             return data_H_T.reset_index(drop=True) , data_P.reset_index(drop=True)


#         def nodes_and_edges(df):
#             x = np.array(df.x)
#             y = np.array(df.y)
#             def car_to_cyl_cood(x, y):
#                 r = np.sqrt(x**2 + y**2)
#                 phi = np.arctan2(y, x)
#                 return r , phi

#             r_hits , phi_hits = car_to_cyl_cood(x, y)

#             df["phi"] = phi_hits
#             df["r"] = r_hits

#             z = np.array(df.z)    
#             r = np.array(df.r)
#             phi = np.array(df.phi)
#             particles = np.array(df.particle_id)
#             hit_id = np.array(df.hit_id)


#             @njit(error_model='numpy')
#             def func(x, y, r, phi, z, particles, hit_id):
#                 node_to_node_feat = []
#                 for i, idx in enumerate(range(len(phi))):
#                     for k, kdx in enumerate(range(len(phi))):
#                         dpdr = (phi[k]- phi[i])/(r[k]-r[i])
#                         dr = r[k]-r[i]
#                         z0 =  z[i] - r[i]*((z[k]-z[i])/(r[k]-r[i]))
#                         if abs(z0) <= 70 and dr > 0:
#                             node_to_node_feat.append([(particles[i] == particles[k])*1, dpdr, idx, kdx, hit_id[idx], hit_id[kdx], dr, z[idx]
#                                           , z[kdx], r[idx], r[kdx], phi[idx], phi[kdx], x[idx], x[kdx], y[idx], y[kdx], z0])
#                 return node_to_node_feat

#             node_to_node_feat = func(x, y, r, phi, z, particles, hit_id) 

#             node_to_node_feat = pd.DataFrame(node_to_node_feat, columns=['Y_k' , "dpdr" , 'node_1', 'node_2',"hit_id_node1","hit_id_node2","dr","z_node1","z_node2","r_node1","r_node2","phi_node1","phi_node2",
#                                     "x_node1","x_node2","y_node1","y_node2","z0"])
#             return node_to_node_feat


#         def REWMH(data): #removing_edges_with_to_many_hits_ids
#             data_sorted_True = data.loc[data.Y_k == 1]
#             data_sorted_True = data_sorted_True.sort_values(by = ["hit_id_node1","dr"],ascending=True)
#             data_True = data_sorted_True.drop_duplicates(subset = "hit_id_node1",ignore_index = True)
#             data_sorted_False = data.loc[data.Y_k == 0]
#             data_sorted_False = data_sorted_False.sort_values(by = ["hit_id_node1","dr"],ascending=True)
#             data_False = data_sorted_False.drop_duplicates(subset = "hit_id_node1",ignore_index = True)
#             True_and_false = pd.concat([data_True, data_False], ignore_index=True)
#             return True_and_false

#         events_called = event_nr
#         volumn_list = detector_sections 
#         pt_cut_start = pt_values[0]
#         pt_cut_end = pt_values[1]
#         data_path = "pages/Data/train_100_events/"

#         st.write(f"Reading data for {events_called} events:")
#         data_H_T , data_P = reading_data(data_path, events_called)
        
        
#         st.write(f"Cleaning data for {events_called} events:")
#         data_H_T_list = []
#         data_P_list = []
#         for __ in stqdm(range(events_called)):
#             data_1 , data_2 = full_data_clean(data_H_T[__], data_P[__], volumn_list, pt_cut_start, pt_cut_end)
#             data_H_T_list.append(data_1)
#             data_P_list.append(data_2)

#         st.write(f"Finding nodes for all {events_called} events:")
#         node_to_node_feat = [nodes_and_edges(data_H_T_list[i]) for i in stqdm(range(events_called))]

#         st.write(f"Splitting the data to {math.floor(len(node_to_node_feat)*0.95)} traning events:")
#         All_edges_TF_traning = [REWMH(node_to_node_feat[:math.floor(len(node_to_node_feat)*0.95)][i]) for i in stqdm(range(math.floor(len(node_to_node_feat)*0.95)))]
#         All_edges_TF_traning_combined = pd.concat(All_edges_TF_traning, ignore_index=True)
        
#         st.write(f"Splitting the data to {math.ceil(len(node_to_node_feat)*0.05)} testing events:")
#         All_edges_TF_testing = [REWMH(node_to_node_feat[-math.ceil(len(node_to_node_feat)*0.05):][i]) for i in stqdm(range(math.ceil(len(node_to_node_feat)*0.05)))]
#         All_edges_TF_testing_combined = pd.concat(All_edges_TF_testing, ignore_index=True)
        
        
#         def MLP_func(data_train, data_test, hidden_layer, max_iter, solver, activation, learning_rate):
#             X_train = data_train[["z_node1","z_node2","r_node1","r_node2","phi_node1","phi_node2","x_node1","x_node2","y_node1","y_node2"]].copy()
#             Y_train = data_train.Y_k
#             X_test = data_test[["z_node1","z_node2","r_node1","r_node2","phi_node1","phi_node2","x_node1","x_node2","y_node1","y_node2"]].copy()
#             Y_test = data_test.Y_k
#             model = MLPClassifier(hidden_layer_sizes = hidden_layer, random_state=1, max_iter = max_iter, solver = solver, activation = activation, learning_rate = learning_rate)
#             model.fit(X_train,Y_train)
#             prop_train = model.predict(X_train)
#             score_train = model.score(X_train, Y_train)
#             prop_test = model.predict(X_test)
#             score_test = model.score(X_test, Y_test)
#             return model, X_train, Y_train, X_test, Y_test, prop_train, score_train, prop_test, score_test
        
#         st.write("Traning and testing of the GNN/MLP has started please wait...")

#         model, X_train, Y_train, X_test, Y_test, prop_train, score_train, prop_test, score_test = MLP_func(All_edges_TF_traning_combined, All_edges_TF_testing_combined, hidden_layers, max_iter, solver, activation, learning_rate)
    
#         st.write("Traning and testing of the GNN/MLP has ended! :)")
        
#         st.write("Here you can see the results of the traning, by using sklean.metrics.classification_report:")
#         # st.write("With MLP parameters set to:")
#         # st.write(f"hidden_layers:", {hidden_layers}, " ,", "max_iter:", {max_iter}, " ,", "solver:", {solver}, " ,", "activation:", {activation}, " ,", "learning_rate:", {learning_rate})
#         report_train = classification_report(Y_train, prop_train, output_dict=True)
#         st.text('Model Report:\n ' + classification_report(Y_train, prop_train))
        
#         st.write("Here you can see the results of the testing, by using sklean.metrics.classification_report:")
#         report_train = classification_report(Y_test, prop_test, output_dict=True)
#         st.text('Model Report:\n ' + classification_report(Y_test, prop_test))
        
        
#         st.write("Plotting the tracks of particles that have over 10 hits, from the test :")
#         truedf = X_test.loc[Y_test == 1]
#         x_node1 = np.array(truedf.x_node1)
#         x_node2 = np.array(truedf.x_node2)
#         y_node1 = np.array(truedf.y_node1)
#         y_node2 = np.array(truedf.y_node2)
#         z_node1 = np.array(truedf.z_node1)
#         z_node2 = np.array(truedf.z_node2)

#         @njit(error_model='numpy')
#         def segment_finder(x_node1, x_node2, y_node1, y_node2, z_node1, z_node2):
#             segment = []
#             for i, idx in enumerate(range(len(x_node1))):
#                 for k, kdx in enumerate(range(len(x_node1))):
#                     if i <= 2500 and z_node2[i] == z_node1[k] and y_node2[i] == y_node1[k] and x_node2[i] == x_node1[k]:
#                         segment.append([idx, kdx])
#                     if i > 2500 and z_node2[k] == z_node1[i] and y_node2[k] == y_node1[i] and x_node2[k] == x_node1[i]:
#                         segment.append([kdx, idx])

#             seg1 = []
#             for i in range(len(segment)):
#                 for k in range(len(segment)):
#                     if segment[i][1] == segment[k][0]:
#                         seg1.append([segment[i][0], segment[i][1], segment[k][1]])

#             seg2 = []
#             for i in range(len(seg1)):
#                 for k in range(len(seg1)):
#                     if seg1[i][2] == seg1[k][0]:
#                         seg2.append([seg1[i][0],seg1[i][1], seg1[k][0],seg1[k][1],seg1[k][2]])
#             seg3 = []
#             for i in range(len(seg2)):
#                 for k in range(len(seg2)):
#                     if seg2[i][4] == seg2[k][0]:
#                         seg3.append([seg2[i][0], seg2[i][1], seg2[i][2], seg2[i][3],seg2[k][0], seg2[k][1],seg2[k][2],seg2[k][3],seg2[k][4]])

#             return seg1 , seg2 , seg3
        
#         seg1 , seg2 , seg3 = segment_finder(x_node1, x_node2, y_node1, y_node2, z_node1, z_node2)
    
#         col1, col2 = st.columns(2)
#         with col1:
#             fig = plt.figure()
#             for i in seg3:
#                 plt.plot(truedf.x_node1[i], truedf.y_node1[i], color = 'blue', alpha = 0.3)
#                 plt.scatter(truedf.x_node1[i], truedf.y_node1[i], color = 'blue', s = 2)
#                 plt.scatter(truedf.x_node2[i], truedf.y_node2[i], s = 15)
#                 plt.plot(truedf.x_node2[i], truedf.y_node2[i])
#             plt.xlabel('x [mm]')
#             plt.ylabel('y [mm]')
#             plt.title(f"""Tracks of {len(seg3)} particles with 10 hits""", fontsize = 20)
#             st.pyplot(fig)
###---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------###     
if page_info == "View detector setup":
    det = pd.read_csv('pages/Data/detectors.csv')

    view_detector = '<p style="font-family:sans-serif; color:White; font-size: 22px;"> View detector sections: </p>'
    st.markdown(view_detector, unsafe_allow_html=True)    
    col1, col2 , col3, col4, col5, col6 = st.columns(6)
    with col1:
        view_detector = st.select_slider("view_detector", ["No", "Yes"], label_visibility = "hidden")
    
    if view_detector == "No":
        st.write("Wating to view detector sections...")
    
    if view_detector == "Yes":    
        detector_volumn = Image.open("pages/Images/Detector volume id.png")
        st.image(detector_volumn,
                width=650,
                output_format="PNG")
    
    def remove_volume_ids(data_H_T, volumn_list):
        data_H_T = data_H_T.loc[data_H_T['volume_id'].isin(volumn_list)]
        return data_H_T

    st.header("Selecet which sections/volume_id you would like to look at:")
    detector_sections = st.multiselect(
        "Select Countrys:",
        options= [7, 8, 12, 13, 14, 16, 17, 18],
        default= [8, 13, 17],
        label_visibility="hidden")
    
    layers = remove_volume_ids(det, detector_sections)
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure(1)
        ax = plt.axes(projection='3d')
        #ax.set_facecolor('xkcd:black')
        ax.set_title(f"Detector volumes {detector_sections}")#.set_position([.5, .002])
        ax.set_xlabel('X (mm)', rotation = 0)
        ax.set_ylabel('Y (mm)', rotation = 0)
        ax.set_zlabel('Z (mm)', rotation = 0)
        ax.scatter3D(layers.cz, layers.cx, layers.cy, c = layers.cy)
        st.pyplot(fig)
    
    
