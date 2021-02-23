import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import geopy.distance


#IMPORT RAW DATABASES
immigration = pd.read_csv("../24290bd.csv", sep = "\t", encoding = "ISO-8859-1")


economy = pd.read_csv("../economy.csv", thousands=',')

spain_economy = pd.read_csv("../spain economy.csv")

distances = pd.read_csv("../worldcities.csv")

conflict = pd.read_csv("../ConflictRecurrenceDatabase.csv")

# Rename columns, cleaning 
    # Immigration
immigration.rename(columns = {"País de origen": "Country", 
                              "Period": "Year",
                              "País de nacimiento": "Country of birth"
                              }, inplace=True)
          
immigration.dropna(axis=0,inplace=True)

immigration["Year"] = pd.to_datetime(immigration.Year, format='%Y').dt.to_period('Y')




    # Economy (all countries)
economy.rename(columns = {"Country Name": "Country", 
                          "Time": "Year", 
                          "GDP (current US$) [NY.GDP.MKTP.CD]": "GDP (current US$)",
                          "GDP per capita (current US$) [NY.GDP.PCAP.CD]": "GDP per capita (current US$)",
                          "GDP growth (annual %) [NY.GDP.MKTP.KD.ZG]": "GDP growth (annual %)",
                          "Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]": "Inflation, consumer prices (annual %)",
                          "GDP per capita, PPP (current international $) [NY.GDP.PCAP.PP.CD]": "GDP per capita, PPP", 
                          "Unemployment, total (% of total labor force) (national estimate) [SL.UEM.TOTL.NE.ZS]": "Unemployment (national estimate)",
                          "Unemployment, total (% of total labor force) (modeled ILO estimate) [SL.UEM.TOTL.ZS]": "Unemployment (modeled ILO estimate)"
                          }, inplace=True)
economy.dropna(axis=0,inplace=True)

economy.replace('..','', inplace=True) 	

economy["Year"] = pd.to_datetime(economy["Year"], format='%Y').dt.to_period('Y')


    # Spanish economy
spain_economy.dropna(axis=0,inplace=True)
spain_economy.drop(columns = ['Series Code'], inplace=True)
spain_economy.drop(columns = ['2020 [YR2020]'], inplace=True)

spain_economy.rename(columns = {'2008 [YR2008]': '2008',
                                '2009 [YR2009]': '2009',
                                '2010 [YR2010]': '2010', 
                                '2011 [YR2011]': '2011', 
                                '2012 [YR2012]': '2012', 
                                '2013 [YR2013]': '2013', 
                                '2014 [YR2014]': '2014', 
                                '2015 [YR2015]': '2015', 
                                '2016 [YR2016]': '2016',
                                '2017 [YR2017]': '2017', 
                                '2018 [YR2018]': '2018', 
                                '2019 [YR2019]': '2019',
                                }, inplace=True)

                                
spain_economy = spain_economy.transpose()

spain_economy.columns = spain_economy.iloc[0]
spain_economy = spain_economy[1:]

spain_economy.rename(columns = {'GDP per capita growth (annual %)': '[ESP]GDP per capita growth (annual %)',
                                'GDP (current US$)': '[ESP]GDP (current US$)',
                                'GDP per capita (current US$)': '[ESP]GDP per capita (current US$)',
                                'GDP per capita, PPP (current international $)': '[ESP]GDP per capita, PPP (current international $)',
                                'Inflation, consumer prices (annual %)': '[ESP]Inflation, consumer prices (annual %)',
                                'Unemployment, total (% of total labor force) (national estimate)': '[ESP]Unemployment, total (% of total labor force) (national estimate)' ,
                                'Unemployment, total (% of total labor force) (modeled ILO estimate)': '[ESP]Unemployment, total (% of total labor force) (modeled ILO estimate)', 
                                }, inplace=True)

spain_economy.drop(['Country Name', 'Country Code'], axis=0, inplace=True)

spain_economy['Year'] = spain_economy.index

spain_economy["Year"] = pd.to_datetime(spain_economy["Year"], format='%Y').dt.to_period('Y')


    # Distances from country of origin to spain
distances.drop(columns = ['city_ascii', 'admin_name', 'id', 'iso2'], inplace=True)

capitals = distances.loc[distances['capital'] == 'primary']

capitals.reset_index(drop=True, inplace = True)

capitals['Distance to Spain[KM]'] = [geopy.distance.distance([capitals['lat'][x], capitals['lng'][x]], [float(40.4189), float(-3.6919)]).km for x in range(len(capitals))]

capitals.drop(columns=['iso3','capital'],inplace=True)
capitals.rename(columns={'country': 'Country'}, inplace=True)

    # Conflicts
conflict.drop(columns = ['country','conflict_new_id', 'conflict_name', 'dyad_new_id', 'dyad_name',
       'type_of_violence','active_year','conflict_dyad',
       'dyad_ep_start', 'dyad_ep_end', 'dyad_recurrence_date', 'dyad_ep_id','conf_recurrence_date', 'conf_ep_id',
       'incompatibility', 'territory_name', 'type_of_conflict',
       'region', 'conf_subid', 'conf_recurrence', 'confactor_recurrence',
       'dyad_recurrence', 'factions', 'link_type_1', 'link_id_1',
       'link_type_2', 'link_id_2', 'link_type_3', 'link_id_3', 'link_type_4',
       'link_id_4', 'link_type_5', 'link_id_5', 'link_type_6', 'link_id_6',
       'dyad_recurrence_years', 'conf_recurrence_years', 'conf_ep_low',
       'conf_ep_best', 'conf_ep_high', 'dyad_ep_low', 'dyad_ep_best',
       'dyad_ep_high', 'conf_ep_freq', 'dyad_ep_freq', 'cont_dyad',
       'cont_conf'], inplace=True)




conflict.rename(columns = {'conf_ep_start':'start', 'conf_ep_end':'end', 'location':'Country'}, inplace=True)
conflict.dropna(axis=0,inplace=True)
conflict.reset_index(drop=True, inplace = True)


conflict["start"] = pd.to_datetime(conflict["start"])
conflict["end"] = pd.to_datetime(conflict["end"])


conflict["start"] = pd.DatetimeIndex(conflict["start"]).year
conflict["end"] = pd.DatetimeIndex(conflict["end"]).year




conflict['start'] = conflict['start'].astype('int64')
conflict['end'] = conflict['end'].astype('int64')
conflict['Year'] = conflict.apply(lambda x : list(range(x['start'],x['end'] + 1)),axis = 1)
conflict.drop(columns=['start','end'], inplace = True)
conflict = conflict.explode('Year')

conflict['Conflict'] = 1

conflict["Year"] = pd.to_datetime(conflict["Year"], format='%Y').dt.to_period('Y')



# MERGE 
complete = pd.merge(immigration , economy,how='inner', on=['Country','Year'])


# CLEAN MERGED DF

condition = complete["Country of birth"].eq("País de origen")
complete["Country of birth"] = np.where(condition, complete["Country"], complete["Country of birth"])

condition2 = complete["Country of birth"].eq("Otros")

complete["Country of birth"] = np.where(condition2, "Other", complete["Country of birth"])


# MERGE ECONOMY + MIGRATION + SPANISH DATA
df =pd.merge(complete, spain_economy, how='inner', on='Year')


# CLEAN MERGED DF

df.loc[df['Country of birth'] == 'Spain', 'Country of Birth_Spanish'] = 1
df.loc[df['Country of birth'] != 'Spain', 'Country of Birth_Spanish'] = 0
df.dropna(axis=0, inplace=True)


Spanish_speaking = ['Mexico',
                    'Honduras',
                    'Nicaragua', 
                    'Dominican Republic', 
                    'Argentina', 
                    'Bolivia',
                    'Colombia',
                    'Chile',
                    'Ecuador',
                    'Paraguay',
                    'Peru',
                    'Uruguay']

df['Spanish_speaking'] = df['Country']
df['Spanish_speaking'].loc[df['Spanish_speaking'].isin(Spanish_speaking)] = 1
df['Spanish_speaking'].loc[df['Spanish_speaking'] != 1] = 0


#  Drop columns that we don't need

df.drop(['Five-year age group', 'Time Code'], axis=1, inplace=True)
# Change column values to 0,1,2
df.loc[df['Nationality'] == 'Spanish ', 'Nationality'] = 1

df.loc[df['Nationality'] == 'Extranjera', 'Nationality'] = 0

df.loc[df['Sex'] == 'Males', 'Sex'] = 0
df.loc[df['Sex'] == 'Females', 'Sex'] = 1

df.rename(columns={"Sex": "Sex_Female", "Nationality": "Nationality_Spanish"}, inplace=True)

df['Nationality_Spanish'] = df['Nationality_Spanish'].astype(int)
df['Sex_Female'] = df['Sex_Female'].astype(int)

#bool to string
booleans = ['Nationality_Spanish', 'Sex_Female']
for column in booleans:
    df[column] = df[column].astype(str)


df.replace(',','', regex=True, inplace=True)

# Change datatype to float for economic indicators

num_columns = df.columns.tolist()
to_num_cols=num_columns[7:]

for column in to_num_cols:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df['Total'] = pd.to_numeric(df['Total'], errors='coerce')
df["Year"]= pd.to_datetime(df["Year"], format='%Y').dt.to_period('Y')
df.drop(columns=['Unemployment (national estimate)', '[ESP]Unemployment, total (% of total labor force) (national estimate)'], inplace=True)
df.drop(df[df.Country == 'Cuba'].index, inplace=True)


# CONVERT COUNTRY INTO CATEGORY
LE = LabelEncoder()
df['Country_code'] = LE.fit_transform(df['Country'])
df['Country_code'] = df['Country_code'].astype(int)

# ADD DISTANCE TO SPAIN AND CONFLICTS

df = pd.merge(df, capitals, how='inner', on='Country')
df = pd.merge(df, conflict, how='left', on=['Country','Year'])

df['Conflict'] = df['Conflict'].replace(np.nan, 0)




#  Add dummy for GDP greater and smaller than Spain

df['Greater_GDP'] = (df['GDP (current US$)'] > df['[ESP]GDP (current US$)']).astype(int)
df['Smaller_GDP'] = (df['GDP (current US$)'] < df['[ESP]GDP (current US$)']).astype(int)




# SPLITTING DF 
Greater_GDP, Smaller_GDP= [x for _, x in df.groupby(df['GDP (current US$)'] < df['[ESP]GDP (current US$)'])]

# DROPPING SPANISH INDICATORS' COLUMNS
Greater_GDP.drop(columns = ['[ESP]GDP per capita growth (annual %)',
                            '[ESP]GDP (current US$)',
                            '[ESP]GDP per capita (current US$)',
                            '[ESP]GDP per capita, PPP (current international $)', 
                            '[ESP]Inflation, consumer prices (annual %)',
                            '[ESP]Unemployment, total (% of total labor force) (modeled ILO estimate)'], inplace=True)


Smaller_GDP.drop(columns = ['[ESP]GDP per capita growth (annual %)',
                            '[ESP]GDP (current US$)',
                            '[ESP]GDP per capita (current US$)',
                            '[ESP]GDP per capita, PPP (current international $)', 
                            '[ESP]Inflation, consumer prices (annual %)',
                            '[ESP]Unemployment, total (% of total labor force) (modeled ILO estimate)'], inplace=True)






Smaller_GDP['Nationality_Spanish'] = Smaller_GDP['Nationality_Spanish'].astype(int)
Smaller_GDP['Sex_Female'] = Smaller_GDP['Sex_Female'].astype(int)





# GROUPED BY YEAR AND COUNTRY, GENERAL DF: TOTAL COLUMN HAS TO BE A SUM, NOT A MEAN.
grouped_df = df.groupby(['Country','Year']).mean()

total_sum = df.groupby(['Country','Year'])['Total'].sum()

total_sum.rename({"Total": "Total_sum"}, inplace=True) 

grouped_df = pd.merge(grouped_df , total_sum,how='inner', on=['Country','Year'])

grouped_df.drop(columns = ['Total_x', 'Country of Birth_Spanish','Country_code',
 'lat',
 'lng',
 'population',], inplace=True)

grouped_df.rename(columns={'Total_y':'Total'}, inplace=True)

list(grouped_df.columns)
grouped_clean = grouped_df[['Unemployment (modeled ILO estimate)',
 '[ESP]Unemployment, total (% of total labor force) (modeled ILO estimate)',
 'Spanish_speaking',
 'Distance to Spain[KM]',
 'Conflict',
 'Greater_GDP',
 'Smaller_GDP',
 'Total']].copy()


grouped_clean['Greater_Unemployment'] = (grouped_clean['Unemployment (modeled ILO estimate)'] > grouped_clean['[ESP]Unemployment, total (% of total labor force) (modeled ILO estimate)']).astype(int)
grouped_clean['Smaller_Unemployment'] = (grouped_clean['Unemployment (modeled ILO estimate)'] < grouped_clean['[ESP]Unemployment, total (% of total labor force) (modeled ILO estimate)']).astype(int)

grouped_clean.drop(columns = ['Unemployment (modeled ILO estimate)', '[ESP]Unemployment, total (% of total labor force) (modeled ILO estimate)'], inplace=True)



# ---------------REGRESSIONS---------------------

print('Only looking at economic factors')
print('')
X = grouped_df[['GDP (current US$)',
                'GDP growth (annual %)',
                'GDP per capita (current US$)',
                'Unemployment (modeled ILO estimate)'
                ]]
Y = grouped_df["Total"]
X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())
print('')
print('')
print('Adding Spanish-Speaking')
print('')
X = grouped_df[['GDP (current US$)',
                'GDP per capita (current US$)', 
                'GDP growth (annual %)',
                'Spanish_speaking',
        ]]
Y = grouped_df["Total"]
X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())
print('')
print('')
print('Keeping only GDP and Adding Distance')
print('')
X = grouped_df[['Spanish_speaking', 
                'Distance to Spain[KM]',
                'Smaller_GDP',
                'Conflict'
                ]]
Y = grouped_df["Total"]
X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())







X = grouped_clean[['Spanish_speaking', 
                'Distance to Spain[KM]',
                'Greater_GDP',
                'Greater_Unemployment',
                'Conflict'
                ]]
Y = grouped_df["Total"]
X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())
print('')
print('')

X = grouped_clean[['Spanish_speaking',
                'Smaller_GDP',
                'Greater_Unemployment',
                'Conflict'
                ]]
Y = grouped_clean["Total"]
X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())




# the p value smaller than 0.05 means that we can reject the null hypothesis
# which means that the variable does have an impact on the dependent variable
# with a 95% confidence.







