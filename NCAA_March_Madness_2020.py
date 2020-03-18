#!/usr/bin/env python
# coding: utf-8

# # 2020 March Madness Machine Learning Competition

# 
# **The goal of this project use data from previous NCAA seasons in order to predict the likelihood, in a percentage, that one team would defeat another in the tournament for the current year using machine learning algorithms. I am using the environment provided Jupyter Notebooks which is very convenient in some ways but only allows Python code. In order use SQL I use a pre-packaged library called sqlite3. The syntax is slightly different that T-SQL that is used in the Microsoft environment used within the VA but still very similar.**

# **First I must import all of the different libraries for python.**

# In[2]:


import pandas as pd                #for data manipulation
import numpy as np                 #linear algebra
import matplotlib.pyplot as plt    #graphs
import seaborn as sns              #graphs
import sklearn as skl              #machine learning      
pd.set_option('max_columns', 35)
pd.set_option('max_rows', 2000)
import warnings
warnings.simplefilter('ignore')


# ***
# **Now to input all the different data sets that I used. There were serveral more than these but ended up not using
# them. Then taking a look at the different data sets to see what information they actually have.**
# ***
# 

# In[129]:


events2015 = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MEvents2015.csv')
events2016 = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MEvents2016.csv')
events2017 = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MEvents2017.csv')
events2018 = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MEvents2018.csv')
events2019 = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MEvents2019.csv')
compact_results = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
detailed_results = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')
seed_round_slots = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneySeedRoundSlots.csv')
tourney_seeds = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneySeeds.csv')
tourney_seeds_1 = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneySeeds.csv')
tourney_slots = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneySlots.csv')
reg_compact = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
reg_detailed = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')
teams = pd.read_csv('/Users/martinacoyne/Downloads/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')


# In[4]:


events2015.head()


# In[5]:


compact_results.head()


# In[6]:


detailed_results.head()


# In[7]:


seed_round_slots.head()


# In[8]:


tourney_seeds.head()


# In[9]:


tourney_slots


# In[10]:


reg_compact.head()


# In[11]:


reg_detailed.head()


# In[12]:


teams.head()


# ***
# **Being that the jupyter notebook interface only runs python in this shell. They have built a special library called sqlite3 in order to code in SQL as well. The below code sets the necessary functions in order to create tables and query them using SQL. It creates the database march.db to create tables within.**
# ***

# In[13]:


import sqlite3 
conn = sqlite3.connect('march.db')
def run_query(q):
    with sqlite3.connect('march.db') as conn:
        return pd.read_sql(q, conn)

def run_command(c):
    with sqlite3.connect('march.db') as conn:
        conn.execute('PRAGMA foreign_keys = ON;')
        conn.isolation_level = None
        conn.execute(c)


# **There are several tables that I want to create immediately**

# In[20]:


teams.to_sql('teams', conn) #'teams' is the name of the table within the database, march.db
reg_detailed.to_sql('reg_detailed', conn)
tourney_seeds.to_sql('tourney_seeds', conn)
compact_results.to_sql('compact_results', conn)
tourney_slots[tourney_slots['Season'].isin(range(2015,2020))].to_sql('slots',conn)


# In[156]:


seeds = tourney_seeds #I need to get a dataset of the seeds without the letters so I can make it an integer and thus, use it in the ML algorithm
for each in range(len(seeds)):
    seeds['Seed'].loc[each] = seeds['Seed'].loc[each].replace('W','').replace('Y','').replace('X','').replace('Z','').replace('a','').replace('b','')
    a = seeds['Seed'].loc[each]
    seeds['Seed'].loc[each] = int(a)
    
seeds.to_sql('seeds',conn) #Then create a table from that filtered dataset to query


# In[157]:


run_query(
'''
SELECT * FROM seeds LIMIT 10
'''
)


# ***
# **The season record was the first statistic that I wanted to include in my set of features used to the predict the outcome of the game. This was done from detailed stats for the regular season (reg_deatailed). In order to get both wins and losses for each team, I had to create two views from the same table, then combine. The result was slightly flawed because Kentucky had no losses in year 2015 so it produced a NaN value for that year. Thus, I had to manually for that particular value.**
# ***

# In[25]:


season_record = run_query(
'''
WITH 

first AS (

    SELECT
        r.Season,
        r.WTeamID TeamID,
        t.Teamname,
        COUNT(r.WTeamID) Wins
    FROM reg_detailed r
    INNER JOIN teams t on t.TeamID = r.WTeamID 
    GROUP BY 2,1
    ),

second AS (
    SELECT
        r.Season,
        r.LTeamID TeamID,
        t.TeamName,
        COUNT(r.LTeamID) Losses
    FROM reg_detailed r
    LEFT JOIN teams t on t.TeamID = r.LTeamID -- LEFT JOIN required because Kentucky had 0 losses. 
    GROUP BY 2,1
    )
    
SELECT 
    f.Season,
    f.TeamID,
    f.TeamName,
    f.Wins,
    s.Losses,
    ROUND(CAST(f.Wins as float)/(CAST(f.Wins as float) + CAST(s.Losses as float)),2) Record
FROM first f
LEFT JOIN second s ON f.TeamID = s.TeamID AND f.Season = s.Season 
'''
)

season_record.head(10)


# In[26]:


season_record['Losses'].loc[2302] = 0 #Kentucky had no losses so I must manually input
season_record['Record'].loc[2302] = 1.0 


# In[33]:


#With those values corrected, I created a new table, 'record', to be later joined with other features.

season_record[season_record['Season'].isin(range(2015,2020))].to_sql('record', conn)


# ***
# **Next I wanted to include many of the statistics from the reg_detailed table again. Since the statistics were listed by Winning team and Losing team, in order to get the entire seasons totals from both games each team won and lost, I had to create two views from the same table, join them by a union. Then finally query that third table and group them by team to include the stats from the games won and the games lost.**
# ***

# In[34]:


season_team_totals = run_query(
'''
WITH

first AS (
    SELECT
        r.Season,
        r.WTeamID TeamID,
        t.TeamName,
        SUM(r.WFGM) FieldGoalsMade,
        SUM(r.WFGA) FieldGoalsAttempted,
        SUM(r.WFGM3) ThreePointMade, 
        SUM(r.WFGA3) ThreePointAttemped, 
        SUM(r.WFTM) FreeThrowsMade, 
        SUM(r.WFTA) FreeThrowsAttempted, 
        SUM(r.WOR) OffensiveRebounds, 
        SUM(r.WDR) DefensiveRebounds,
        SUM(r.WAst) Assists, 
        SUM(r.WTO) TurnOvers, 
        SUM(r.WStl) Steals, 
        SUM(r.WBlk) Blocks, 
        SUM(r.WPF) PersonalFouls
    FROM reg_detailed r
    INNER JOIN teams t ON t.TeamID = r.WTeamID
    GROUP BY 1,2
    ),

second AS (
    SELECT
        r.Season,
        r.LTeamID TeamID,
        t.TeamName,
        SUM(r.LFGM) FieldGoalsMade,
        SUM(r.LFGA) FieldGoalsAttempted,
        SUM(r.LFGM3) ThreePointMade, 
        SUM(r.LFGA3) ThreePointAttemped, 
        SUM(r.LFTM) FreeThrowsMade, 
        SUM(r.LFTA) FreeThrowsAttempted, 
        SUM(r.LOR) OffensiveRebounds, 
        SUM(r.LDR) DefensiveRebounds,
        SUM(r.LAst) Assists, 
        SUM(r.LTO) TurnOvers, 
        SUM(r.LStl) Steals, 
        SUM(r.LBlk) Blocks, 
        SUM(r.LPF) PersonalFouls
    FROM reg_detailed r
    INNER JOIN teams t ON t.TeamID = r.LTeamID
    GROUP BY 1,2
),

third AS (
SELECT * FROM first

UNION

SELECT * FROM second
)

SELECT 
    Season,
    TeamID,
    TeamName,
    SUM(FieldGoalsMade) FieldGoalsMade,
    SUM(FieldGoalsAttempted) FieldGoalsAttempted,
    SUM(ThreePointMade) ThreePointMade,          
    SUM(ThreePointAttemped) ThreePointAttemped, 
    SUM(FreeThrowsMade) FreeThrowsMade, 
    SUM(FreeThrowsAttempted) FreeThrowsAttempted, 
    SUM(OffensiveRebounds) OffensiveRebounds, 
    SUM(DefensiveRebounds) DefensiveRebounds,
    SUM(Assists) Assists, 
    SUM(TurnOvers) TurnOvers, 
    SUM(Steals) Steals, 
    SUM(Blocks) Blocks, 
    SUM(PersonalFouls) PersonalFouls
From third
WHERE Season = 2015 OR Season = 2016 OR Season = 2017 OR Season = 2018 OR Season = 2019
GROUP BY 1, 2

    
''')
season_team_totals.head(10)


# In[36]:


season_team_totals.to_sql('season_team_totals', conn) #Then committed this back to a table


# ***
# **I looked at the stats for the teams of many of the seasons and created graphs to visualize it (not shown in this notebook because it was lengthy). From that analysis I saw that the offensive stats really stuck out for those teams that made it the final four. Thus I started looking at Three Point shot made, 2 point shots made, and free throws. Then looking at the percentage of them. Then I created one statistic from all which I played around with to see which was most accurate in predicting who would win those games leading to the final four. It is:**
# 
# ### %3-Points + 2***(%2-Points) + 3***(%Free-Throws)
# ***
# 
# 
# 

# In[39]:


run_command(
'''
CREATE VIEW total AS
SELECT 
    Season,
    TeamID,
    CAST(FieldGoalsMade as float)/CAST(FieldGoalsAttempted as float) FieldGoalPer,
    CAST(ThreePointMade as float)/CAST(ThreePointAttemped as float) ThreePer,
    CAST(FreeThrowsMade as float)/CAST(FreeThrowsAttempted as float) FreePer,
    (CAST(ThreePointMade as float)/CAST(ThreePointAttemped as float)) + (2*(CAST(FieldGoalsMade as float)/CAST(FieldGoalsAttempted as float))) + (3*(CAST(FreeThrowsMade as float)/CAST(FreeThrowsAttempted as float))) Total
FROM season_team_totals  
GROUP BY 1,2 ORDER BY 1,2
'''
)


# In[40]:


run_query(
'''
SELECT * FROM total
'''
).head()


# ***
# **Next I wanted to look at upsets. The total number of upsets by year and especially for the first two rounds, since 1985, what is the percentage of upsets for each of the seeds. I considered using this percentages for just the first two rounds. However, it did not prove to be more effective than the ML algorithm I created.**
# *** 

# In[57]:


upsets = run_query(
'''
WITH

first AS (
    SELECT 
        c.Season,
        c.DayNum,
        c.WTeamID,
        t.TeamName WTeamName,
        s.Seed WSeed,
        c.WScore, 
        c.LTeamID, 
        c.LScore
    FROM compact_results c
    LEFT JOIN seeds s ON c.WTeamID = s.TeamID AND c.Season = s.Season
    INNER JOIN teams t ON c.WTeamID = t.TeamID
    ),

second AS (
    SELECT
        f.Season,
        f.DayNum,
        f.WTeamID,
        f.WTeamName,
        f.WSeed,
        f.WScore, 
        f.LTeamID,
        t.TeamName LTeamName,
        s.Seed LSeed,
        f.LScore
    FROM first f
    LEFT JOIN seeds s ON f.LTeamID = s.TeamID AND f.Season = s.Season
    INNER JOIN teams t on f.LTeamID = t.TeamID
)

SELECT * FROM second
WHERE DayNum > 135
''')


# In[60]:


upsets.to_sql('upsets', conn)


# In[61]:


upsets_by_year = run_query(
'''
SELECT 
    Season,
    COUNT(Season) NumberUpsets
FROM upsets_1
GROUP BY 1
''')
upsets_by_year.head()


# In[62]:


upsets_by_year.plot(x='Season', y='NumberUpsets', kind='bar', figsize=(25,10))


# In[63]:


upset_first_round = run_query(
'''
SELECT
    WSeed,
    LSeed,
    COUNT(WSeed) TimesUpset
FROM upsets_1
WHERE DayNum = 136 OR DayNum = 137
GROUP BY 1 ORDER BY 1 DESC
''')
upset_first_round['Percentage'] = upset_first_round['TimesUpset']/136 
upset_first_round


# In[64]:


second_round_upsets = run_query(
'''
SELECT
    WSeed,
    LSeed,
    COUNT(WSeed) NumberUpsets
FROM upsets_1
WHERE DayNum = 138 or DayNum = 139
GROUP BY 1,2
'''
)
second_round_upsets['Percentage'] = second_round_upsets['NumberUpsets']/136
second_round_upsets


# **The next statistic I wanted to add was Points per possesion. The equation I used to determine possession is stated below.** 
# ### Possessions = field goal attempts – offensive rebounds + turnovers + (0.475 x free throw attempts)

# In[68]:


APP = run_query(
'''

WITH

    first AS (
        SELECT 
            *,
            CAST(WFGA AS float) - CAST(WOR AS float) + CAST(WTO AS float) + (.475 * CAST(WFTA AS float)) WPossessions,
            CAST(LFGA AS float) - CAST(LOR AS float) + CAST(LTO AS float) + (.475 * CAST(LFTA AS float)) LPossessions
        FROM reg_detailed
        ),
    
    second as (
        SELECT 
            Season,
            WTeamID,
            AVG(WScore) AvgWScore,
            AVG(WPossessions) Winning,
            AVG(WScore)/AVG(WPossessions) WPointsPerPoss
        FROM first 
        GROUP BY 1,2 ORDER BY 1
        ),
        
    third AS (
        SELECT
            Season,
            LTeamID,
            AVG(LScore) AvgLScore,
            AVG(LPossessions) Losing,
            AVG(LScore)/AVG(LPossessions) LPointsPerPoss
        FROM first
        GROUP BY 1,2 ORDER BY 1
        ),
        
    fourth AS (
        SELECT
            s.Season,
            s.WTeamID TeamID,
            te.TeamName,
            s.Winning,
            t.Losing,
            (s.Winning + t.Losing)/2 AvgPoss,
            (s.AvgWScore + t.AvgLScore)/2 AvgPoints,
            (s.AvgWScore + t.AvgLScore)/(s.Winning + t.Losing) AvgPointsPerPoss
        FROM second s
        LEFT JOIN third t ON s.WTeamID = t.LTeamID AND s.Season = t.Season
        INNER JOIN teams te ON te.TeamID = s.WTeamID
        WHERE s.Season = 2019 or s.Season = 2018 or s.Season = 2017 or s.Season = 2016 or s.Season = 2015
        ORDER BY 1
        )
        
    SELECT * FROM fourth
'''
)
APP['AvgPointsPerPoss'].loc[137] = 1.135870 #Once again with Kentucky having no losses in 2015, producted NaN value.
APP.head(10)


# In[71]:


APP.to_sql('points_per_poss',conn)


# ***
# **Next I wanted to look at the events datasets. These are datasets that contain every single event that happened in every game during the regular season which results to several million rows of data. Using only certain events for those games, I created a score for each team. I compared these with the seed rankings for several years and it actually matches up quite well. Using SQL for this estimate was a huge time saver. Last year I used a FOR loop in python to accumulate the scores for each and it would 15 - 30 minutes to run the code. With SQL, less than 1 minute.**
# ***

# In[73]:


events2015 = events2015[events2015['EventType'].isin(['reb', 'made2', 'assist', 'steal','made3', 'made1', 'block'])]
events2016 = events2016[events2016['EventType'].isin(['reb', 'made2', 'assist', 'steal','made3', 'made1', 'block'])]
events2017 = events2017[events2017['EventType'].isin(['reb', 'made2', 'assist', 'steal','made3', 'made1', 'block'])]
events2018 = events2018[events2018['EventType'].isin(['reb', 'made2', 'assist', 'steal','made3', 'made1', 'block'])]
events2019 = events2019[events2019['EventType'].isin(['reb', 'made2', 'assist', 'steal','made3', 'made1', 'block'])]
events = pd.concat([events2015,events2016,events2017,events2018,events2019])
events.to_sql('events', conn)


# In[76]:


run_command(
'''
CREATE VIEW scores AS
SELECT
    e.Season,
    e.EventTeamID TeamID,
    t.TeamName,
    COUNT(e.EventTeamID) Score
FROM events e
INNER JOIN teams t ON t.TeamID = e.EventTeamID
GROUP BY 1,2
''')


# In[77]:


run_query(
'''
SELECT * FROM scores_1 LIMIT 10
''')


# ***
# **The next feature I wanted to look at was not just how many wins and losses each team had but how much they won by and how much they lost by. I did that by creating a score using the avg points won by and lost by for all teams for that season. Below is the equation I used.**

# ### Calc = (AvgPointsWonBy - YearAvg) + (YearAvg - AvgPointsLostBy)
# ***

# In[78]:


reg_detailed_2015 = reg_detailed[reg_detailed['Season'] == 2015] 
reg_detailed_2016 = reg_detailed[reg_detailed['Season'] == 2016]
reg_detailed_2017 = reg_detailed[reg_detailed['Season'] == 2017]
reg_detailed_2018 = reg_detailed[reg_detailed['Season'] == 2018]
reg_detailed_2019 = reg_detailed[reg_detailed['Season'] == 2019]
reg_detailed_2015['AvgPointsWonBy'] = reg_detailed_2015['WScore'] - reg_detailed_2015['LScore']
reg_detailed_2016['AvgPointsWonBy'] = reg_detailed_2016['WScore'] - reg_detailed_2016['LScore']
reg_detailed_2017['AvgPointsWonBy'] = reg_detailed_2017['WScore'] - reg_detailed_2017['LScore']
reg_detailed_2018['AvgPointsWonBy'] = reg_detailed_2018['WScore'] - reg_detailed_2018['LScore']
reg_detailed_2019['AvgPointsWonBy'] = reg_detailed_2019['WScore'] - reg_detailed_2019['LScore']
avg_2015 = reg_detailed_2015['AvgPointsWonBy'].mean()
avg_2016 = reg_detailed_2016['AvgPointsWonBy'].mean()
avg_2017 = reg_detailed_2017['AvgPointsWonBy'].mean()
avg_2018 = reg_detailed_2018['AvgPointsWonBy'].mean()
avg_2019 = reg_detailed_2019['AvgPointsWonBy'].mean()


# In[82]:


reg_detail_full = pd.concat([reg_detailed_2015,reg_detailed_2016,reg_detailed_2017,reg_detailed_2018,reg_detailed_2019]).reset_index(drop=True)
reg_detail_full.to_sql('reg_detail',conn) #This combines all years together and creates a table


# In[85]:


calc = run_query(
'''

WITH
first as (
        SELECT
            r.Season,
            r.WTeamID WTeamID,
            t.TeamName WTeamName,
            AVG(r.AvgPointsWonBy) AvgPointsWonBy
        FROM reg_detail r
        INNER JOIN teams t on t.TeamID = r.WTeamID
        GROUP BY 1,2 ORDER BY 1
    ),
    
    second AS (
        SELECT
            r.Season,
            r.LTeamID LTeamID,
            t.TeamName LTeamName,
            AVG(r.AvgPointsWonBy) AvgPointsLostBy
        FROM reg_detail r
        INNER JOIN teams t on t.TeamID = r.LTeamID
        GROUP BY 1,2 ORDER BY 1
        ),
    
    third AS (
        SELECT 
            f.*,
            s.AvgPointsLostBy
        FROM first f
        LEFT JOIN second s ON s.LTeamID = f.WTeamID AND f.Season = s.Season
        ),
        
    fourth AS(
        SELECT
            Season,
            AVG(AvgPointsWonBy) YearAvg
        FROM reg_detail
        GROUP BY 1
    ),
    
    fifth AS (
    SELECT 
        t.*,
        (t.AvgPointsWonBy - f.YearAvg) + (f.YearAvg - t.AvgPointsLostBy) Calc
    FROM third t
    LEFT JOIN fourth f ON t.Season = f.Season
    )
    
    SELECT * FROM fifth
'''
)

calc['AvgPointsLostBy'].loc[137] = 0      #Once again, Kentucky not having losses
calc['Calc'].loc[137] = 20.941176         #Once again, Kentucky not having losses 
calc.head()


# In[91]:


calc.to_sql('calc',conn)


# ***
# **Lastly, there are many statistics based on the teams performance but there is no way to assess the competition. If a team won by 20 points each time but played horrible teams, it is very misleading. I had to go outside of the provided datasets by Kaggle for this. KenPom.com has this statistic, strength of schedule. But it took quite a bit of cleaning up to make the dataset compatible with the ones I was using.**
# ***
# 

# In[95]:


pom_15 = pd.read_csv('/Users/martinacoyne/Downloads/pom_2015.csv')
pom_16 = pd.read_csv('/Users/martinacoyne/Downloads/pom_2016.csv')
pom_17 = pd.read_csv('/Users/martinacoyne/Downloads/pom_2017.csv')
pom_18 = pd.read_csv('/Users/martinacoyne/Downloads/pom_2018.csv')
pom_19 = pd.read_csv('/Users/martinacoyne/Downloads/pom_2019.csv')
pom_15['Season'] = 2015
pom_16['Season'] = 2016
pom_17['Season'] = 2017
pom_18['Season'] = 2018
pom_19['Season'] = 2019
pom = pd.concat([pom_15,pom_16,pom_17,pom_18,pom_19]).reset_index(drop=True)
pom = pom.rename(columns={'AdjEM.1': 'SchedStrength'})


# In[96]:


pom.head()


# In[97]:


#the names of the teams must cleaned up so that we can combine the POM dataset with our current datasets

for each in range(len(pom)):
    a = pom['Team'].loc[each]
    b = a[:-2]
    if a[-1] in ['0','1','2','3','4','5','6','7','8','9']:
        pom['Team'].loc[each] = b
    else: 
        pass
for each in range(len(pom)):
    a = pom['Team'].loc[each]
    b = a[:-1]
    if a[-1] in [' ']:
        pom['Team'].loc[each] = b
    else: 
        pass
for each in range(len(pom)):
    a = pom['Team'].loc[each]
    b = a[:-1]
    if a[-1] in ['.']:
        pom['Team'].loc[each] = b
    else: 
        pass


# In[98]:


#Now that we have the names cleaned up, we have to make sure they all match. I will correct this with a dictionary
#and using the map() function

mapping_dict2 = {
 'Abilene Christian':'Abilene Chr',
 'Albany':'SUNY Albany',
 'Arkansas Little Rock':'Ark Little Rock',
 'Cal St. Bakersfield':'CS Bakersfield',
 'Cal St. Fullerton':'CS Fullerton',
 'Coastal Carolina':'Coastal Car',
 'College of Charleston':'Col Charleston',
 'East Tennessee St':'ETSU',
 'Eastern Washington':'E Washington',
 'Fairleigh Dickinson':'F Dickinson',
 'Florida Gulf Coast':'FL Gulf Coast',
 'Green Bay':'WI Green Bay',
 'Kent St':'Kent',
 'Loyola Chicago':'Loyola-Chicago',
 'Middle Tennessee':'MTSU',
 "Mount St. Mary's":"Mt St Mary's",
 'N.C. State':'NC State',
 'North Carolina Central':'NC Central',
 'North Dakota St':'N Dakota St',
 'Northern Kentucky':'N Kentucky',
 'Prairie View A&M':'Prairie View',
 "Saint Joseph's":"St Joseph's PA",
 'Saint Louis':'St Louis',
 "Saint Mary's":"St Mary's CA",
 'South Dakota St':'S Dakota St',
 'Southern':'Southern Univ',
 'St. Bonaventure':'St Bonaventure',
 "St. John's": "St John's",
 'Stephen F. Austin':'SF Austin',
 'Texas Southern':'TX Southern'
 }


# In[99]:


for each in range(len(pom)): #now use this mapping dict to get all the names correct.
    a = pom['Team'].loc[each]
    if a in mapping_dict2.keys():
        pom['Team'].loc[each] = mapping_dict2[a]
    else:
        pass


# In[101]:


pom.to_sql('pom',conn)


# In[102]:


run_query(
'''
SELECT * FROM pom LIMIT 10
'''
)


# ***
# **Now I have all the data points I want for the different teams in the different seasons but in order to create the ML algorithm, it has to be in right order for which team played which for all the seasons 2015 - 2019. This is actually quite complicated to do with how datasets are arranged. Thus, I wrote a function in python to help do all the leg work arranging it for all the different seasons.**
# ***

# In[130]:


def make_tourney_bracket(year):   
    compact = compact_results[compact_results['Season'] == year].reset_index(drop=True)
    seeds = tourney_seeds_1[tourney_seeds_1['Season'] == year]
    slots = tourney_slots[tourney_slots['Season'] == year]
    slots['Winner'] = slots['WeakSeed']
    slots['DayNum'] = 2020

    seedsDict = {}
    for each in seeds.index:
        a = seeds['Seed'].loc[each]
        b = seeds['TeamID'].loc[each]
        seedsDict[a] = b

    prelim = slots.iloc[63:]
    round1 = slots.iloc[0:32]
    round2 = slots.iloc[32:48]
    round3 = slots.iloc[48:56]
    round4 = slots.iloc[56:60]
    round5 = slots.iloc[60:62]
    round6 = slots.iloc[62]

    prelim['StrongSeed'] = prelim['StrongSeed'].map(seedsDict)
    prelim['WeakSeed'] = prelim['WeakSeed'].map(seedsDict)

    prelimWinners = compact['WTeamID'].loc[:4].tolist()
    winPrelim = {}

    for each in prelim.index:
        prelim['DayNum'].loc[each] = 134
        a = prelim['StrongSeed'].loc[each]
        b = prelim['WeakSeed'].loc[each]
        c = prelim['Slot'].loc[each]
        if a in prelimWinners:
            winPrelim[c] = a
            prelim['Winner'].loc[each] = a
        else:
            winPrelim[c] = b
            prelim['Winner'].loc[each] = b

    seedsDict.update(winPrelim)

    round1['StrongSeed'] = round1['StrongSeed'].map(seedsDict)
    round1['WeakSeed'] = round1['WeakSeed'].map(seedsDict)

    round1Winners = compact['WTeamID'].loc[4:35].tolist()
    winDict1 = {}

    for each in round1.index:
        round1['DayNum'].loc[each] = 136
        a = round1['StrongSeed'].loc[each]
        b = round1['WeakSeed'].loc[each]
        c = round1['Slot'].loc[each]
        if a in round1Winners:
            winDict1[c] = a
            round1['Winner'].loc[each] = a
        else:
            winDict1[c] = b
            round1['Winner'].loc[each] = b

    round2['StrongSeed'] = round2['StrongSeed'].map(winDict1)
    round2['WeakSeed'] = round2['WeakSeed'].map(winDict1)

    round2Winners = compact['WTeamID'].loc[36:51].tolist()
    winDict2 = {}

    for each in round2.index:
        round2['DayNum'].loc[each] = 138
        a = round2['StrongSeed'].loc[each]
        b = round2['WeakSeed'].loc[each]
        c = round2['Slot'].loc[each]
        if a in round2Winners:
            winDict2[c] = a
            round2['Winner'].loc[each] = a
        else:
            winDict2[c] = b
            round2['Winner'].loc[each] = b

    round3['StrongSeed'] = round3['StrongSeed'].map(winDict2)
    round3['WeakSeed'] = round3['WeakSeed'].map(winDict2)

    round3Winners = compact['WTeamID'].loc[52:59].tolist()
    winDict3 = {}

    for each in round3.index:
        round3['DayNum'].loc[each] = 143
        a = round3['StrongSeed'].loc[each]
        b = round3['WeakSeed'].loc[each]
        c = round3['Slot'].loc[each]
        if a in round3Winners:
            winDict3[c] = a
            round3['Winner'].loc[each] = a
        else:
            winDict3[c] = b
            round3['Winner'].loc[each] = b

    round4['StrongSeed'] = round4['StrongSeed'].map(winDict3)
    round4['WeakSeed'] = round4['WeakSeed'].map(winDict3)

    round4Winners = compact['WTeamID'].loc[60:63].tolist()
    winDict4 = {}

    for each in round4.index:
        round4['DayNum'].loc[each] = 145
        a = round4['StrongSeed'].loc[each]
        b = round4['WeakSeed'].loc[each]
        c = round4['Slot'].loc[each]
        if a in round4Winners:
            winDict4[c] = a
            round4['Winner'].loc[each] = a
        else:
            winDict4[c] = b
            round4['Winner'].loc[each] = b

    round5['StrongSeed'] = round5['StrongSeed'].map(winDict4)
    round5['WeakSeed'] = round5['WeakSeed'].map(winDict4)

    round5Winners = compact['WTeamID'].loc[64:66].tolist()
    winDict5 = {}

    for each in round5.index:
        round5['DayNum'].loc[each] = 152
        a = round5['StrongSeed'].loc[each]
        b = round5['WeakSeed'].loc[each]
        c = round5['Slot'].loc[each]
        if a in round5Winners:
            winDict5[c] = a
            round5['Winner'].loc[each] = a
        else:
            winDict5[c] = b
            round5['Winner'].loc[each] = b


    round6['StrongSeed'] = winDict5['R5WX']
    round6['WeakSeed'] = winDict5['R5YZ']
    a = compact.index[-1]
    round6Winners = compact['WTeamID'].loc[a]

    total = pd.concat([prelim,round1,round2,round3,round4,round5])

    return total


# **Then I will execute the function for each year and add them all together to make one table.**

# In[131]:


tourn2015 = make_tourney_bracket(2015)
tourn2016 = make_tourney_bracket(2016)
tourn2017 = make_tourney_bracket(2017)
tourn2018 = make_tourney_bracket(2018)
tourn2019 = make_tourney_bracket(2019)
thetourney = pd.concat([tourn2015,tourn2016,tourn2017,tourn2018,tourn2019])


# In[134]:


thetourney.head()


# In[137]:


thetourney.to_sql('thetourney',conn)


# ***
# **OK, so now I have all my data points a table with the correct structure to add them to. Time to add them all together. SQL is brilliant for this type of task. I also added a column using CASE which determines if the strong team was in fact the winner. This column will be used in the prediction.**
# ***

# In[158]:


ultimate = run_query(
'''
WITH
    
    first AS(
        SELECT
            tt.Season,
            tt.DayNum,
            tt.StrongSeed StrongTeamID,
            t.TeamName StrongName,
            se.Seed StrongSeed,
            tt.WeakSeed,
            tt.Winner
        FROM thetourney tt
        INNER JOIN teams t ON t.TeamID = tt.StrongSeed
        INNER JOIN seeds se ON tt.StrongSeed = se.TeamID AND tt.Season = se.Season
        GROUP BY 1,2,3
        ),

    second AS(
        SELECT
            f.Season,
            f.DayNum,
            f.StrongTeamID,
            f.StrongName,
            f.StrongSeed,
            f.WeakSeed WeakTeamID,
            t.TeamName WeakName,
            se.Seed WeakSeed,
            f.Winner
        FROM first f
        INNER JOIN teams t ON t.TeamID = f.WeakSeed
        INNER JOIN seeds se ON f.WeakSeed = se.TeamID AND f.Season = se.Season
        GROUP BY 1,2,3
        ),
        
    third AS (
     
        SELECT
            s.Season,
            s.DayNum,
            s.StrongTeamID,
            s.StrongName,
            s.StrongSeed,
            c.Calc StrongCalc,
            sc.Score StrongScore,
            p.AvgPointsPerPoss StrongAvgPPP,
            t.Total StrongTotal,
            r.Record StrongRecord,
            pom.SchedStrength StrongStrength,
            s.WeakTeamID,
            s.WeakName,
            s.WeakSeed,
            s.Winner
        FROM second s
        LEFT JOIN calc c ON s.StrongTeamID = c.WTeamID AND s.Season = c.Season
        LEFT JOIN scores_1 sc ON s.StrongTeamID = sc.TeamID AND s.Season = sc.Season
        LEFT JOIN points_per_poss p ON p.TeamID = s.StrongTeamID AND p.Season = s.Season
        LEFT JOIN total t ON s.StrongTeamID = t.TeamID AND s.Season = t.Season
        LEFT JOIN record r ON r.TeamID = s.StrongTeamID AND r.Season = s.Season
        LEFT JOIN pom ON pom.Team = s.StrongName AND s.Season = pom.Season
        )
        
        SELECT
            th.Season,
            th.DayNum,
            th.StrongTeamID,
            th.StrongName,
            th.StrongSeed,
            th.StrongCalc,
            th.StrongScore,
            th.StrongAvgPPP,
            th.StrongTotal,
            th.StrongRecord,
            th.StrongStrength,
            th.WeakTeamID,
            th.WeakName,
            th.WeakSeed,
            c.Calc WeakCalc,
            sc.Score WeakScore,
            p.AvgPointsPerPoss WeakAvgPPP,
            t.total WeakTotal,
            r.Record WeakRecord,
            pom.SchedStrength WeakStrength,
            th.Winner WinnerID,
            te.TeamName WinnerName,
            CASE
            WHEN th.Winner = th.StrongTeamID THEN 1
            ELSE 0
            END
            AS Outcome
        FROM third th
        LEFT JOIN calc c ON th.WeakTeamID = c.WTeamID AND th.Season = c.Season
        LEFT JOIN scores_1 sc ON th.WeakTeamID = sc.TeamID AND th.Season = sc.Season
        LEFT JOIN points_per_poss p ON p.TeamID = th.WeakTeamID AND p.Season = th.Season
        LEFT JOIN total t ON th.WeakTeamID = t.TeamID AND th.Season = t.Season
        LEFT JOIN record r ON r.TeamID = th.WeakTeamID AND r.Season = th.Season
        LEFT JOIN pom ON pom.Team = th.WeakName AND th.Season = pom.Season
        LEFT JOIN teams te ON th.Winner = te.TeamID
    
            
'''
)


# In[159]:


ultimate.head(10)


# ***
# # Machine Learning Section
# ***

# In[162]:


from sklearn.tree import DecisionTreeRegressor 


# 
# **NCAA_features are the designated features for this particular algorithm. The other lines of code set it up according to the features and result.**

# In[164]:


ncaa_features = ['StrongSeed',
       'StrongCalc', 'StrongScore', 'StrongAvgPPP', 'StrongTotal',
       'StrongRecord', 'StrongStrength', 'WeakSeed',
       'WeakCalc', 'WeakScore', 'WeakAvgPPP', 'WeakTotal', 'WeakRecord',
       'WeakStrength']
y = ultimate.Outcome             #the outcome it is trying to predict    
X = ultimate[ncaa_features]
ncaa_model = DecisionTreeRegressor(random_state=1)
ncaa_model.fit(X,y)


# ***
# **The below code is train the algorithm on other data in order to improve it.**
# ***

# In[168]:


from sklearn.metrics import mean_absolute_error

predicted_ncaa_outcomes = ncaa_model.predict(X)
mean_absolute_error(y, predicted_ncaa_outcomes)

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

ncaa_model = DecisionTreeRegressor()

ncaa_model.fit(train_X, train_y)

val_predictions = ncaa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


# **And finally the prediction itself**

# In[169]:


Z = ultimate[ncaa_features]
prediction = ncaa_model.predict(Z)

ultimate['Prediction'] = prediction


# In[170]:


ultimate.head(15)


# ***
# **To check the accuracy of it based on the test data. So we can see it predicted the correct team with about 91.5% accurcy.**
# ***

# In[171]:


x = 0
for each in ultimate.index:
    a = ultimate['Outcome'].loc[each]
    b = ultimate['Prediction'].loc[each]
    if a == b:
        x += 1
print(x)
print(x/330)


# ***
# **Now I will also set up a Random Forest algorithm to compare.**
# ***

# In[174]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


# In[175]:


Z = ultimate[ncaa_features]
prediction_1 = forest_model.predict(Z)

ultimate['Forrest'] = prediction_1


# In[177]:


ultimate.head(15)


# In[ ]:




