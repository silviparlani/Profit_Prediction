import numpy as np
import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from nums_from_string import nums_from_string
from sklearn import linear_model
from streamlit_pandas_profiling import st_profile_report
from streamlit_option_menu import option_menu
from PIL import Image


dataset=pd.read_csv('cars_dataset.csv')
#selecting required columns
use_dataset=dataset[['Make','PriceInLakhs','Cylinders','Drivetrain','FuelTankCapacity','BodyType','ARAICertifiedMileage','KerbWeight','Gears','GroundClearance','Power','Torque','SeatingCapacity','Type','Wheelbase','GrossVehicleWeight','ProfitPercent','UnitsSoldPerThousand']]
use_dataset.to_csv(r'C:\Users\muska\Desktop\Engage\Cars_Profit_Prediction_Dataset')
ds=pd.read_csv('Cars_Profit_Prediction_Dataset')
ds.drop(['Unnamed: 0'],axis=1,inplace=True)

#CLEANING THE DATASET
import warnings
warnings.filterwarnings("ignore")

#removing letters from numeric data
ds['KerbWeight'] = ds['KerbWeight'].str.replace(r"[a-zA-Z]",'')
ds["KerbWeight"] = ds["KerbWeight"].astype("float64")
ds['Wheelbase'] = ds['Wheelbase'].str.replace(r"[a-zA-Z]",'')
ds["Wheelbase"] = ds["Wheelbase"].astype("float64")
ds['GrossVehicleWeight'] = ds['GrossVehicleWeight'].str.replace(r"[a-zA-Z]",'')
ds["GrossVehicleWeight"] = ds["GrossVehicleWeight"].astype("float64")
ds['GroundClearance'] = ds['GroundClearance'].str.replace(r"[a-zA-Z]",'')
ds["GroundClearance"] = ds["GroundClearance"].astype("float64")
ds['FuelTankCapacity'] = ds['FuelTankCapacity'].str.replace(r"[a-zA-Z]",'')
ds["FuelTankCapacity"] = ds["FuelTankCapacity"].astype("float64")
ds["Gears"] = ds["Gears"].astype("float64")
ds['ARAICertifiedMileage'] = ds['ARAICertifiedMileage'].str.extract(r'(\d+.\d+)').astype('float')

#standardizing power and torque
def one_rpm(value):
    number=nums_from_string.get_nums(value)
    for i in range(0,len(number)):
        if number[i]<0:
            number[i]=number[i]*-1
    if len(number)==3:
        average=(number[1]+number[2])/2
        std=number[0]/average
    else:
        std = number[0] / number[1]
    return(std)

li=[]
for i in ds['Power'].tolist():
    li.append(one_rpm(str(i)))
ds.drop(labels='Power',axis=1,inplace=True)
ds['PowerSTD']=li

li1=[]
for i in ds['Torque'].tolist():
    li1.append(one_rpm(str(i)))
ds.drop(labels='Torque',axis=1,inplace=True)
ds['TorqueSTD']=li1

#filling missing numeric values
a = ds.select_dtypes('float64')
ds[a.columns] = a.fillna(a.mean())

#dropping rows with null values for catagorical variables
ds=ds.dropna(how='any',axis=0)

numeric_columns = ds.select_dtypes('float64')
categorical_columns=ds.select_dtypes('object')
car=ds['Make']
categorical_columns.drop(['Make'],inplace=True,axis=1)
ds.drop(['Make'],inplace=True,axis=1)

#removing outliers (determined by observing the boxplot for each column)
ds.drop(ds[ds['FuelTankCapacity'] > 89].index, inplace=True)
ds.drop(ds[ds['ARAICertifiedMileage'] > 100].index, inplace=True)
ds.drop(ds[ds['KerbWeight'] > 2200].index, inplace=True)
ds.drop(ds[(ds['Gears'] < 4) | (ds['Gears'] > 7)].index, inplace=True)
ds.drop(ds[(ds['GroundClearance'] < 130) | (ds['GroundClearance'] > 220)].index, inplace=True)
ds.drop(ds[(ds['Wheelbase'] < 2200) | (ds['Wheelbase'] > 3130)].index, inplace=True)
ds.drop(ds[(ds['GrossVehicleWeight'] < 1400) | (ds['GrossVehicleWeight'] > 2200)].index, inplace=True)
ds.drop(ds[ds['PowerSTD'] > 0.085].index, inplace=True)
ds.drop(ds[ds['TorqueSTD'] > 0.35].index, inplace=True)
#****DATASET CLEANED****

target = 'ProfitPercent'
st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.markdown('''
# Automobile Data Analysis
# 
''')
st.sidebar.image("main_menu_image.png", use_column_width=True)
st.sidebar.markdown('''# ''')
with st.sidebar:
    rad=option_menu(
        menu_title="Main Menu",
        #menu_icon= "graph-up-arrow",
        options=["Explore Dataset","Visualize Data","Predict Profit"],
        icons=["clipboard-data","pie-chart-fill","graph-up"],
        #orientation="horizontal"
    )


if rad=="Explore Dataset":
    st.title("Exploration")
    st.markdown('''
    ## Automobile Industry Dataset
    ## Pandas Profile Report
    ### Overview
    ''')
    st.image("overview.png")
    st.markdown('''### Variables''')
    st.image("bodytype.png")
    st.image("cylinders.png")
    st.image("drivetrain.png")
    st.image("ftc.png")
    st.image("gears.png")
    st.image("groundclearance.png")
    st.image("gvw.png")
    st.image("kerbweight.png")
    st.image("mileage.png")
    st.image("power.png")
    st.image("price.png")
    st.image("seatingcapacity.png")
    st.image("torque.png")
    st.image("type.png")
    st.image("units.png")
    st.image("wheelbase.png")
    st.markdown('''### Correlation''')
    st.image("correlation.png")
elif rad=="Visualize Data":
    st.title("Visualization")
    st.markdown('''## Select Type of Visualization''')
    rad1=st.selectbox("Select",["Distribution of Target Variable","Distribution of Other Variables","Relationship Between Columns and Target Variable"])
    if rad1=="Distribution of Target Variable":
        st.markdown('''
            ## Distribution of Target Variable
            ### Target Variable: Profit Percent
            ''')
        plt.title('Profit Percent Distribution Plot')
        sns.distplot(ds[['ProfitPercent']])
        st.pyplot()
        plt.title('Profit Percent Spread')
        sns.boxplot(y=ds['ProfitPercent'])
        st.pyplot()
    elif rad1=="Distribution of Other Variables":
        st.markdown('''
                    ## Distribution of Other Variables
                    ''')
        car.value_counts(normalize=True).plot(kind='bar', color='green')
        plt.title("Distribution of Car Brand ")
        st.pyplot()
        for col in ds.columns:
            if col=='ProfitPercent':
                continue
            elif col=='UnitsSoldPerThousand':
                sns.distplot(ds[col])
            elif col in numeric_columns.columns:
                s="Distribution of "+col
                plt.title(s)
                sns.distplot(ds[col])
            elif col in categorical_columns.columns:
                s="Distribution of "+col
                plt.title(s)
                ds[col].value_counts(normalize=True).plot(kind='bar', color='green')
                plt.xticks(rotation=45)
            st.pyplot()
    else:
        st.markdown('''
                    ## Relationship between Target Variable and Other Variables
                    ### Target Variable: Profit Percent
                    ''')
        target = 'ProfitPercent'
        numeric_columns.drop(['ProfitPercent'], inplace=True, axis=1)
        data = ds.iloc[0:ds.shape[0]:10]
        sns.barplot(x=target, y=car, data=ds, ci=0)
        plt.title("Car Brand vs. Target")
        st.pyplot()
        for column in numeric_columns:
            sns.regplot(x=target, y=column, data=data, scatter_kws={"color": "pink"}, line_kws={"color": "black"})
            s = column+" vs Target"
            plt.title(s)
            st.pyplot()
        for column in categorical_columns:
            sns.barplot(x=target, y=column, data=ds, ci=0,palette=("Greens_r"))
            s = column + " vs Target"
            plt.title(s)
            st.pyplot()
else:
    st.title("Prediction")
    st.markdown('''
                        # Profit Prediction
                        ## Select Features
                        ''')
    Y = ds[target].values
    X = ds.drop(['ProfitPercent'], axis=1)
    # encoding categorical columns
    # using frequency encoding
    unencoded = X[['Drivetrain', 'BodyType', 'Type']]
    BodyType_Dict = X['BodyType'].value_counts()
    Type_Dict = X['Type'].value_counts()
    Drivetrain_Dict = X['Drivetrain'].value_counts()
    X['Encoded_BodyType'] = X['BodyType'].map(BodyType_Dict)
    X['Encoded_Type'] = X['Type'].map(Type_Dict)
    X['Encoded_Drivetrain'] = X['Drivetrain'].map(Drivetrain_Dict)
    X.drop(['BodyType', 'Drivetrain', 'Type'], inplace=True, axis=1)
    regressor = linear_model.LinearRegression()
    regressor.fit(X, Y)

    # order:'PriceInLakhs', 'Cylinders', 'FuelTankCapacity', 'ARAICertifiedMileage',
    # 'KerbWeight', 'Gears', 'GroundClearance', 'SeatingCapacity','Wheelbase',
    # 'GrossVehicleWeight', 'UnitsSoldPerThousand', 'PowerSTD','TorqueSTD',
    # 'Encoded_BodyType', 'Encoded_Type', 'Encoded_Drivetrain'
    features=[]
    price=st.slider("Price In Lakhs",min_value=3.0,max_value=20.0,step=0.1)
    st.write("Selected Price: ",price," Lakhs")
    features.append(price)

    cylinder = st.slider("No. of Cylinders", min_value=2, max_value=8,step=2)
    st.write("Selected No. of Cylinders: ", cylinder)
    features.append(cylinder)

    ftc = st.slider("Fuel Tank Capacity", min_value=20, max_value=80)
    st.write("Selected Fuel Tank Capacity: ", ftc, "litres")
    features.append(ftc)

    mileage = st.slider("ARAI Certified Mileage", min_value=6.0, max_value=25.0,step=0.1)
    st.write("Selected Mileage: ", mileage, "km/litre")
    features.append(mileage)

    kw = st.slider("Kerb Weight", min_value=700, max_value=2000)
    st.write("Selected Weight: ", kw, "kg")
    features.append(kw)

    g = st.slider("Gears", min_value=4, max_value=7)
    st.write("Selected No. of Gears: ", g)
    features.append(g)

    gc = st.slider("Ground Clearance", min_value=150, max_value=220)
    st.write("Selected Ground Clearance: ", gc, "mm")
    features.append(gc)

    seating = st.slider("Seating Capacity", min_value=2, max_value=9)
    st.write("Selected Seating Capacity: ", seating)
    features.append(seating)

    wb = st.slider("Wheelbase", min_value=2200, max_value=3000)
    st.write("Selected Wheelbase: ", wb, "mm")
    features.append(wb)

    gvw = st.slider("GrossVehicleWeight", min_value=1500, max_value=2200)
    st.write("Selected Gross Vehicle Weight: ", gvw, "kg")
    features.append(gvw)

    unit = st.slider("UnitsSoldPerThousand")
    st.write("Selected No. of Units: ", unit, "thousand")
    features.append(unit)

    power = st.slider("Power/RPM", min_value=0.002, max_value=0.080, step=0.001)
    st.write("Selected Power: ", power)
    features.append(power)

    torque = st.slider("Torque/RPM", min_value=0.002, max_value=0.200,step=0.001)
    st.write("Selected Torque: ", torque)
    features.append(torque)

    bt=st.selectbox("Body Type",list(ds['BodyType'].value_counts().index))
    btn=0
    if bt=='SUV':
        btn=441
    elif bt=='Sedan':
        btn=333
    elif bt=='Hatchback':
        btn=315
    elif bt=='Coupe':
        btn=41
    elif bt=='MUV':
        btn=39
    elif bt=='MPV':
        btn=30
    elif bt=='Crossover':
        btn=18
    elif bt=='Convertible':
        btn=18
    elif bt=='Crossover, SUV':
        btn=4
    elif bt=='Sports':
        btn=3
    elif bt=='Pick-up':
        btn=3
    elif bt=='Sedan, Coupe':
        btn=2
    elif bt=='Sports, Convertible':
        btn=2
    elif bt=='Sports, Hatchback':
        btn=1
    elif bt=='Sedan, Crossover':
        btn=1
    else:
        btn=1
    features.append(btn)

    t = st.selectbox("Vehicle Type", list(ds['Type'].value_counts().index))
    tn=0
    if t=="Manual":
        tn=714
    elif t=="Automatic":
        tn=510
    elif t=="AMT":
        tn=18
    elif t=="DCT":
        tn=7
    else:
        tn=3
    features.append(tn)

    dt = st.selectbox("Drivetrain", list(ds['Drivetrain'].value_counts().index))
    dtn=0
    if dt=="FWD":
        dtn=877
    elif dt=="RWD":
        dtn=170
    elif dt=="AWD":
        dtn=146
    else:
        dtn=59
    features.append(dtn)

    prediction=regressor.predict([features]).tolist()
    st.markdown('''
    #
    ### Prediction:
    ''')
    st.write("Predicted Profit: ","{:.2f}".format(prediction[0]),"%")