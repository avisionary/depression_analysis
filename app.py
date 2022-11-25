# importing libraries
import streamlit as st
import pandas as pd
import joblib
import altair as alt
import plotly.express as px
import numpy as np
import gzip
#import brotli


import bz2
import pickle
import _pickle as cPickle

def _max_width_(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

_max_width_()

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

# title 
st.header("Depression Analysis")
st.markdown('Predicting if a person is at risk of depression using their personality type and demographics')

# reading in dataset
df = pd.read_csv("data/depression_overall.csv")


# st.text(df.head())

fig_data = pd.read_csv("data/depression_graph.csv")
# fig_data2 = pd.read_csv("data/fig_data.csv")


# st.text(fig_data.head())
col1, col2 = st.columns(2)

# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])


# test altair chart
# c = alt.Chart(chart_data).mark_circle().encode(
#     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

# st.altair_chart(c, use_container_width=True)



#altair main chart
# selection = alt.selection_multi(fields=['demo','label_var'])
# click = alt.selection_multi()
# color = color=alt.Color(
#             "count", scale=alt.Scale(scheme="redyellowblue"), legend=None,
#         )

# hist1 = alt.Chart(fig_data).mark_bar().encode(
#     x='demo',
#     y='count',
#     color=alt.condition(selection, 'label_var', alt.value('grey'),scale=alt.Scale(scheme=""),
#             legend=None),
#     tooltip=[
#             alt.Tooltip("demo", title="demo"),
#             alt.Tooltip("count", title="count"),
#             alt.Tooltip("label_var", title="label_var")],
# ).add_selection(
#      selection
# ).properties(width=300, height=300)

# hist2 = alt.Chart(fig_data2).mark_bar().encode(
#     x='count',
#     y='target',
#     color = alt.Color('label_var',scale=alt.Scale(scheme="yellowgreenblue"),legend=None),
#     tooltip=[alt.Tooltip("target", title="target"),
#             alt.Tooltip("label_var", title="label_var"),
#             alt.Tooltip("demo", title="demo")]  
# ).add_selection(
#      selection
#  ).transform_filter(
#    selection
# ).properties(width=400, height=100)

# total_participation_viz=alt.vconcat(hist1, hist2).properties(
#     title={
#       "text": ["Which Demographic by Label has the most count?"], 
#       "subtitle": ["Click to see Count of Demographic by Depression Target"],
#       "subtitleFontSize":13,
#       'subtitleFontStyle':'italic',
#       "color": "black",
#       "subtitleColor": "black"
#     }
# ).configure(background='#E7E6E1')

# st.altair_chart(hist1)

fig = px.scatter(fig_data, x="target", y="age",color="target")
col1.plotly_chart(fig, use_container_width=True)

fig = px.scatter(fig_data, x="target",y ="education",color="target")
col2.plotly_chart(fig, use_container_width=True)


# personality types
st.subheader("Personality Type Questions")

st.markdown('The following ten personality type questions will be rated "I see myself as:" _____  such that')

# input variables
display_tipi = ["Disagree strongly", "Disagree moderately","Disagree a little","Neither agree nor disagree","Agree a little","Agree moderately","Agree strongly"]

options = list(range(1,len(display_tipi)+1))

col3, col4 = st.columns(2)

with col3:
    TIPI1 = st.selectbox("Extraverted, enthusiastic", options, format_func=lambda x: display_tipi[x-1])
    TIPI2 = st.selectbox("Critical, quarrelsome", options, format_func=lambda x: display_tipi[x-1])
    TIPI3 = st.selectbox("Dependable, self-disciplined", options, format_func=lambda x: display_tipi[x-1])
    TIPI4 = st.selectbox("Anxious, easily upset", options, format_func=lambda x: display_tipi[x-1])
    TIPI5 = st.selectbox("Open to new experiences, complex", options, format_func=lambda x: display_tipi[x-1])
    
with col4:
    TIPI6 = st.selectbox("Reserved, quiet", options, format_func=lambda x: display_tipi[x-1])
    TIPI7 = st.selectbox("Sympathetic, warm", options, format_func=lambda x: display_tipi[x-1])
    TIPI8 = st.selectbox("Disorganized, careless", options, format_func=lambda x: display_tipi[x-1])
    TIPI9 = st.selectbox("Calm, emotionally stable", options, format_func=lambda x: display_tipi[x-1])
    TIPI10 = st.selectbox("Conventional, uncreative", options, format_func=lambda x: display_tipi[x-1])

#st.write(tipi1)


st.subheader("Vocabulary Type Questions")
st.markdown("In the grid below, check all the words whose definitions you are sure you know")

display_vic = ["I don't know","I know"]
options_vic = list(range(len(display_vic)))
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    vcl1 = st.radio("boat", options_vic, format_func=lambda x: display_vic[x])
    vcl2 = st.radio("incoherent", options_vic, format_func=lambda x: display_vic[x])
    vcl3 = st.radio("pallid", options_vic, format_func=lambda x: display_vic[x])
    vcl4 = st.radio("robot", options_vic, format_func=lambda x: display_vic[x])
    vcl5 = st.radio("audible", options_vic, format_func=lambda x: display_vic[x])
    vcl6 = st.radio("cuivocal", options_vic, format_func=lambda x: display_vic[x])
    vcl7 = st.radio("paucity", options_vic, format_func=lambda x: display_vic[x])
    vcl8 = st.radio("epistemology", options_vic, format_func=lambda x: display_vic[x])

with col2:
    vcl9 = st.radio("florted", options_vic, format_func=lambda x: display_vic[x])
    vcl10 = st.radio("decide", options_vic, format_func=lambda x: display_vic[x])
    vcl11 = st.radio("pastiche", options_vic, format_func=lambda x: display_vic[x])
    vcl12 = st.radio("verdid", options_vic, format_func=lambda x: display_vic[x])
    vcl13 = st.radio("abysmal", options_vic, format_func=lambda x: display_vic[x])
    vcl14 = st.radio("lucid", options_vic, format_func=lambda x: display_vic[x])
    vcl15 = st.radio("betray", options_vic, format_func=lambda x: display_vic[x])
    vcl16 = st.radio("funny", options_vic, format_func=lambda x: display_vic[x])



st.subheader("Demographic Type Questions")



# education
display_edu = ["Less than high school","High school","University degree","Graduate degree"]
options_edu = list(range(1,len(display_edu)+1))
education = st.selectbox("How much education have you completed?", options_edu, format_func=lambda x: display_edu[x-1])
#st.write(education)
for i in range(len(display_edu)+1):
    exec(f"education_{i} = 0")



# urban
display_urban = ["Rural (country side)","Suburban","Urban (town, city)"]
options_urban = list(range(1,len(display_urban)+1))
urban = st.selectbox("What type of area did you live when you were a child?", options_urban, format_func=lambda x: display_urban[x-1])
#st.write(urban)
for i in range(len(display_urban)+1):
    exec(f"urban_{i} = 0")



# gender
display_gender = ["Male","Female","Other"]
options_gender = list(range(1,len(display_gender)+1))
gender = st.selectbox("What is your gender?", options_gender, format_func=lambda x: display_gender[x-1])
#st.write(gender)
for i in range(len(display_gender)+1):
    exec(f"gender_{i} = 0")


# Age
age = st.number_input("How many years old are you?",min_value=12,max_value=88)
AgeBand_1 = 0
AgeBand_2 = 0
AgeBand_3 = 0
AgeBand_4 = 0


# religion
display_religion = ["Agnostic", "Atheist", "Buddhist", "Christian (Catholic)", "Christian (Mormon)", "Christian (Protestant)", "Christian (Other)", "Hindu", "Jewish", "Muslim","Sikh","Other"]
options_religion = list(range(1,len(display_religion)+1))
religion = st.selectbox("What is your religion?", options_religion, format_func=lambda x: display_religion[x-1])
#st.write(religion)
#st.write(religion + gender)
for i in range(len(display_religion)+1):
    exec(f"religion_{i} = 0")


# sexual orientation
display_orientation = ["Heterosexual", "Bisexual", "Homosexual", "Asexual", "Other"]
options_orientation = list(range(1,len(display_orientation)+1))
orientation = st.selectbox("What is your sexual orientation?", options_orientation, format_func=lambda x: display_orientation[x-1])
#st.write(orientation)
for i in range(len(display_orientation)+1):
    exec(f"orientation_{i} = 0")


# race
display_race = ["Asian", "Arab", "Black", "Indigenous Australian", "Native American","White","Other"]
options_race = list(range(10,(len(display_race)+1)*10,10))
race = st.selectbox("What is your race?", options_race, format_func=lambda x: display_race[int((x-1)/10)])
#st.write(race)
for i in options_race:
    exec(f"race_{i} = 0")


# marital status
display_married = ["Never married", "Currently married", "Previously married"]
options_married = list(range(1,len(display_married)+1))
married = st.selectbox("What is your marital status?", options_married, format_func=lambda x: display_married[x-1])
#st.write(married)
for i in range(len(display_married)+1):
    exec(f"married_{i} = 0")


# family size
familysize = st.number_input("Including you, how many children did your mother have?")

VCL = vcl1+vcl2+vcl3+vcl4+vcl5-vcl6+vcl7+vcl8-vcl9+vcl10+vcl11-vcl12+vcl13+vcl14+vcl15+vcl16

type = st.radio("Do you want to use above data or csv type?",["Above","CSV"])
if type == "CSV":
    txt = st.text_area("Enter comma separated values here:")
    
    
# If button is pressed
if st.button("Submit"):
    if type == "CSV":
        (TIPI1,TIPI2,TIPI3,TIPI4,TIPI5,TIPI6,TIPI7,TIPI8,TIPI9,TIPI10,education,urban,gender,age,religion,orientation,race,married,familysize,VCL) = (2,5,2,2,2,6,5,5,7,2,1,3,2,17,4,3,20,1,3,5)
        #(TIPI1,TIPI2,TIPI3,TIPI4,TIPI5,TIPI6,TIPI7,TIPI8,TIPI9,TIPI10,education,urban,gender,age,religion,orientation,race,married,familysize,VCL) = txt
    age = float(age)
    if age >=12.924 and age < 32.0:
        AgeBand_1 = 1
    if age >=32 and age < 51:
        AgeBand_2 = 1
    if age >=51 and age < 70:
        AgeBand_3 = 1
    if age >=70 and age < 89:
        AgeBand_4 = 1


    exec(f"education_{education} = 1")
    exec(f"urban_{urban} = 1")
    exec(f"gender_{gender} = 1")
    exec(f"religion_{religion} = 1")
    exec(f"orientation_{orientation} = 1")
    exec(f"race_{race} = 1")
    exec(f"married_{married} = 1")

    #Finding the mean and standard deviation of x and normalizing train x 
    u_x_vcl = np.mean(df['VCL'])
    sigma_x_vcl = np.std(df['VCL'])
    VCL = (VCL - u_x_vcl)/sigma_x_vcl

    #Finding the mean and standard deviation of y and normalizing train y 
    u_x_familysize = np.mean(df['familysize'])
    sigma_x_familysize = np.std(df['familysize'])
    familysize = (familysize - u_x_familysize)/sigma_x_familysize

    columns = ['TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8',
       'TIPI9', 'TIPI10', 'familysize', 'VCL', 'education_0', 'education_1',
       'education_2', 'education_3', 'education_4', 'urban_0', 'urban_1',
       'urban_2', 'urban_3', 'gender_0', 'gender_1', 'gender_2', 'gender_3',
       'religion_0', 'religion_1', 'religion_2', 'religion_3', 'religion_4',
       'religion_5', 'religion_6', 'religion_7', 'religion_8', 'religion_9',
       'religion_10', 'religion_11', 'religion_12', 'orientation_0',
       'orientation_1', 'orientation_2', 'orientation_3', 'orientation_4',
       'orientation_5', 'race_10', 'race_20', 'race_30', 'race_40', 'race_50',
       'race_60', 'race_70', 'married_0', 'married_1', 'married_2',
       'married_3', 'AgeBand_1', 'AgeBand_2', 'AgeBand_3', 'AgeBand_4']

    # Store inputs into dataframe
    X = pd.DataFrame([[TIPI1, TIPI2, TIPI3, TIPI4, TIPI5, TIPI6, TIPI7, TIPI8,
       TIPI9, TIPI10, familysize, VCL, education_0, education_1,
       education_2, education_3, education_4, urban_0, urban_1,
       urban_2, urban_3, gender_0, gender_1, gender_2, gender_3,
       religion_0, religion_1, religion_2, religion_3, religion_4,
       religion_5, religion_6, religion_7, religion_8, religion_9,
       religion_10, religion_11, religion_12, orientation_0,
       orientation_1, orientation_2, orientation_3, orientation_4,
       orientation_5, race_10, race_20, race_30, race_40, race_50,
       race_60, race_70, married_0, married_1, married_2,
       married_3, AgeBand_1, AgeBand_2, AgeBand_3, AgeBand_4]], 
                     columns = columns)

    st.markdown("You have entered the following data:")
    st.write(X.head())

    # Unpickle classifier
    #clf = joblib.load("StackedPickle.pkl")
    #clf = joblib.load(brotli.decompress("StackedPickle.pkl.bt"))
    clf = decompress_pickle("StackedPickle_A.pbz2") 

    # Get prediction
    prediction = clf.predict(X)[0]

    # Output prediction
    if prediction == 0:
        st.markdown(f"**This person is NOT at risk of severe depression**")

    if prediction == 1:
        st.markdown(f"**This person is AT risk of severe depression. Take action!**")



    data = [[age,education,prediction]]
    df = pd.DataFrame(data, columns=['age', 'edu','pred'])

    col_g1, col_g2 = st.columns(2)
    fig = px.scatter(fig_data, x="target", y="age",color="target")
    fig.add_traces(
    px.scatter(df, x='pred', y='age').update_traces(marker_size=20, marker_color="red").data
)
    #col1.plotly_chart(fig, use_container_width=True)
    col_g1.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(fig_data, x="target", y="education",color="target")
    fig.add_traces(
    px.scatter(df, x='pred', y='edu').update_traces(marker_size=20, marker_color="red").data
)
    #col2.plotly_chart(fig, use_container_width=True)
    col_g2.plotly_chart(fig, use_container_width=True)
    
    