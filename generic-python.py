# %%
import pandas as pd

# Step 1: Load the data
file_path = '/Users/seyed/Git/Hub/Learning-Style-Questionnaire/Raw_Data.csv'
df = pd.read_csv(file_path)

# Step 2: Show basic info
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())

# Step 3: Check for missing or empty values
missing_values = df.isnull().sum()
print("\nMissing values per column:\n", missing_values)


# %%
# Step 4: Drop unnecessary columns
columns_to_drop = ['Email', 'Name', 'Last modified time']
df_cleaned = df.drop(columns=columns_to_drop)

# Save the cleaned version 
df_cleaned.to_csv('/Users/seyed/Git/Hub/Learning-Style-Questionnaire/Cleaned_Data.csv', index=False)

# %%
# Step 5: Check for exact duplicate rows
duplicate_rows = df_cleaned[df_cleaned.duplicated()]
print("Duplicate rows:")
print(duplicate_rows)


# %%
import pandas as pd


# 1. Rename columns to shorter names
rename_map = {
    'ID': 'ID',
    'Start time': 'Start',
    'Completion time': 'End',
    'Age?': 'Age',
    'Gender?': 'Gender',
    'Year of study?': 'Year',
    'From the scale of 1 to 10 how familiar are you with the concept of learning styles?': 'Familiarity',
    'Which learning style best describes you?': 'PrefStyle',
    'How many hours do you use visual learning aids (e.g. diagram, charts ,videos)?': 'VisualHrs',
    'How many hours do you spend on audio-based learning ( Lectures, podcasts, discussions)?': 'AudioHrs',
    'How many hours do you spend on hands-on or practical learning ( experiments,  case studies)?': 'PracticalHrs',
    'How many hours do you study using written materials (Books, articles, notes)?': 'WrittenHrs',
    'From the scale of 1 to 10 how much LSBU teaching methods support your learning style?': 'SupportRating',
    'Which LSBU learning method do you find most effective?': 'BestMethod',
    "How often do you use LSBU's digital learning resources (Moodle, recorded lectures, online library)?": 'DigitalFreq',
    "How effective do you find LSBU's current teaching methods?": 'MethodEffectiveness',
    'What is your average assessment results(%)?': 'AssessmentCat',
    'From the scale of 1 to 10 how much do you believe  that LSBU have necessary resources matching your learning style? ': 'ResourceRating'
}
df_prep = df_cleaned.rename(columns=rename_map)

# 2. Convert Start/End to elapsed minutes
df_prep['Start'] = pd.to_datetime(df_prep['Start'])
df_prep['End']   = pd.to_datetime(df_prep['End'])
df_prep['ElapsedMins'] = (df_prep['End'] - df_prep['Start']).dt.total_seconds() / 60

# 3. Convert Year of study to numeric
year_map = {
    'First year':  1,
    'Second year': 2,
    'Third year':  3
    }
df_prep['YearNum'] = df_prep['Year'].map(year_map)


# 4. Cast numeric columns
num_cols = ['Age', 'YearNum', 'Familiarity', 'VisualHrs', 'AudioHrs', 'PracticalHrs',
             'WrittenHrs', 'SupportRating', 'DigitalFreq', 'MethodEffectiveness', 'ResourceRating']
df_prep[num_cols] = df_prep[num_cols].apply(pd.to_numeric, errors='coerce')

# 5. Mapping style from long text to short category
pref_map = {
    'Reading/ Writing ( Text-based learning)': 'Reading',
    'Mix of styles':                       'Mixed',
    'Visual (Images, charts)':             'Visual',
    'Auditory ( Listening, discussions)':  'Auditory',
    'Kinesthetic ( Hands-on activities)':  'Kinesthetic'
}
df_prep['PrefStyleShort'] = df_prep['PrefStyle'].map(pref_map)

# 6. Recode AssessmentCat into ordered categories or numeric mid-points
#    For example, map to midpoint values:
mapping = {
    'less than 40': 20,
    '41 to 60':     50,
    '60 to 70':     65,
    'Above 70':     85
}
df_prep['AssessmentScore'] = df_prep['AssessmentCat'].map(mapping)

# 7. Create new features
df_prep['TotalStudyHrs'] = df_prep[['VisualHrs','AudioHrs','PracticalHrs','WrittenHrs']].sum(axis=1)

# Check modal alignment—for each row the max‐hours mode vs. PrefStyle
def dominant_mode(row):
    modes = {
        'Visual': row['VisualHrs'],
        'Auditory': row['AudioHrs'],
        'Practical': row['PracticalHrs'],
        'Written': row['WrittenHrs']
    }
    return max(modes, key=modes.get)

df_prep['DominantMode'] = df_prep.apply(dominant_mode, axis=1)
df_prep['ModeMatchesPreference'] = (df_prep['DominantMode'] == df_prep['PrefStyleShort'])

# 8. Drop redundant columns
# List of columns to remove now that they're redundant
to_drop = [
    'PrefStyle',
    'AssessmentCat',
    'Start',
    'End',
    'Year'
]
# Drop them
df_prep = df_prep.drop(columns=to_drop)


# 9. Save to new CSV (this will become your “Sheet 2” Pre-processed Data)
df_prep.to_csv('/Users/seyed/Git/Hub/Learning-Style-Questionnaire/Preprocessed_Data.csv', index=False)

# 10. Quick check
print("Final shape:", df_prep.shape)
print("Any missing in numeric cols?\n", df_prep[num_cols + ['AssessmentScore']].isnull().sum())
print("Mode vs Pref count:\n", df_prep['ModeMatchesPreference'].value_counts())


# %%
print(df_prep.info())
print(df_prep.describe())

# %%
# Filter to the “False” cases
false_df = df_prep[df_prep['ModeMatchesPreference'] == False]

# Compute the stats
mean_elapsed = false_df['ElapsedMins'].mean()
min_elapsed  = false_df['ElapsedMins'].min()
max_elapsed  = false_df['ElapsedMins'].max()

print(f"Elapsed time when ModeMatchesPreference is False:")
print(f"  Mean: {mean_elapsed:.1f} minutes")
print(f"  Min : {min_elapsed:.1f} minutes")
print(f"  Max : {max_elapsed:.1f} minutes")


# %%
from pandas import read_csv
df_prep = read_csv('/Users/seyed/Git/Hub/Learning-Style-Questionnaire/Preprocessed_Data.csv')


# %%
OneMin_percentile = (df_prep['ElapsedMins'] < 1).mean() * 100
print(f"Percentage of respondents who completed the survey in less than 1 minute: {OneMin_percentile:.2f}%")

# %%
import pandas as pd

# Load your preprocessed data
df = pd.read_csv('/Users/seyed/Git/Hub/Learning-Style-Questionnaire/Preprocessed_Data.csv')

# Select the numeric columns you care about
numeric_cols = [
    'Age', 'YearNum', 'Familiarity', 
    'VisualHrs', 'AudioHrs', 'PracticalHrs', 'WrittenHrs',
    'SupportRating', 'MethodEffectiveness', 'DigitalFreq',
    'ResourceRating', 'AssessmentScore', 'TotalStudyHrs', 'ElapsedMins'
]

# 1. Summary via describe()
desc = df[numeric_cols].describe().T
print("=== Describe (count, mean, std, min, quartiles, max) ===")
print(desc)

# 2. Compute mode for each column
modes = df[numeric_cols].mode().iloc[0]
print("\n=== Mode ===")
print(modes)

# 3. Compute variance for each column
variances = df[numeric_cols].var()
print("\n=== Variance ===")
print(variances)

# 4. Count of non-null and nulls
null_counts = df[numeric_cols].isnull().sum()
print("\n=== Missing Values ===")
print(null_counts)


# %%
# 1. Frequency distributions (value counts & percentages)
for col in ['PrefStyleShort', 'YearNum', 'ModeMatchesPreference']:
    counts = df[col].value_counts(dropna=False)
    pct    = df[col].value_counts(normalize=True, dropna=False) * 100
    print(f"\n--- {col} Value Counts ---")
    print(pd.concat([counts, pct.round(1)], axis=1).rename(columns={col: 'Count', col: 'Percent'}))

# 2. Pivot table: average assessment by style
pivot_style = df.pivot_table(
    index='PrefStyleShort',
    values='AssessmentScore',
    aggfunc=['count', 'mean', 'std']
)
pivot_style.columns = ['N', 'MeanScore', 'StdDevScore']
print("\n--- Assessment Score by Preferred Style ---")
print(pivot_style)

# 3. Pivot table: mean TotalStudyHrs by Year and by ModeMatchesPreference
pivot_year_mode = df.pivot_table(
    index='YearNum',
    columns='ModeMatchesPreference',
    values='TotalStudyHrs',
    aggfunc=['mean', 'count']
)
print("\n--- Total Study Hours by Year & Match Status ---")
print(pivot_year_mode)

# 4. Crosstab: PrefStyleShort vs. ModeMatchesPreference
ct = pd.crosstab(df['PrefStyleShort'], df['ModeMatchesPreference'], margins=True)
print("\n--- Preference vs. Match Crosstab ---")
print(ct)

# 5. Distribution checks: boxplot stats (min, Q1, median, Q3, max)
def box_stats(series):
    q1 = series.quantile(0.25)
    q2 = series.quantile(0.50)
    q3 = series.quantile(0.75)
    return {
        'min': series.min(),
        'Q1':  q1,
        'median': q2,
        'Q3':  q3,
        'max': series.max()
    }

for col in ['VisualHrs', 'AudioHrs', 'PracticalHrs', 'WrittenHrs', 'TotalStudyHrs']:
    stats = box_stats(df[col])
    print(f"\n--- Box stats for {col} ---")
    print(stats)

# 6. Correlation matrix (Pearson’s r)
corr = df[['Familiarity', 'SupportRating', 'MethodEffectiveness', 
           'DigitalFreq', 'TotalStudyHrs', 'AssessmentScore']].corr()
print("\n--- Correlation Matrix ---")
print(corr.round(2))

# 7. Grouped summary: average MethodEffectiveness by digital-use tertiles
df['DigitalTertile'] = pd.qcut(df['DigitalFreq'], q=3, labels=['Low', 'Med', 'High'])
grouped = df.groupby('DigitalTertile')['MethodEffectiveness'].agg(['count', 'mean', 'std'])
print("\n--- MethodEffectiveness by Digital-Use Tertile ---")
print(grouped)


# %%
import matplotlib.pyplot as plt

cols_to_plot = ['VisualHrs', 'AudioHrs', 'PracticalHrs', 'WrittenHrs']
data_to_plot = [df[col].dropna() for col in cols_to_plot]

plt.figure(figsize=(10, 6))
plt.boxplot(data_to_plot, labels=cols_to_plot)
plt.title('Boxplots of Study Hours by Learning Mode')
plt.ylabel('Hours')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# %%
cols = ['Familiarity', 'SupportRating', 'MethodEffectiveness', 
        'DigitalFreq', 'TotalStudyHrs', 'AssessmentScore', 'ElapsedMins']
corr_matrix = df[cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, aspect='equal', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix')
plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
plt.yticks(range(len(cols)), cols)
plt.tight_layout()
plt.show()

# %%
corr = df[['Familiarity', 'SupportRating', 'MethodEffectiveness', 
           'DigitalFreq', 'TotalStudyHrs', 'AssessmentScore']].corr()
print("\n--- Correlation Matrix ---")
print(corr.round(2))


# %%
from scipy.stats import pearsonr

# 1. Filter to matched students
matched = df[df['ModeMatchesPreference'] == True]

# 2. Extract the two series
x = matched['TotalStudyHrs']
y = matched['AssessmentScore']

# 3. Compute Pearson correlation and p-value
r, p = pearsonr(x, y)

print(f"Number of matched students: {len(matched)}")
print(f"Pearson’s r = {r:.2f}")
print(f"p-value     = {p:.3f}")

# %%



