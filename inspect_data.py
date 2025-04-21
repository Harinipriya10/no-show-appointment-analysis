import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('noshowappointments.csv')

# 1. Check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum())

# 2. Remove duplicates
print(f"\nInitial rows: {len(df)}")
df.drop_duplicates(inplace=True)
print(f"Rows after removing duplicates: {len(df)}")

# 3. Standardize text values
df['Gender'] = df['Gender'].str.upper()
df['No-show'] = df['No-show'].str.title()

# 4. Clean column names
df.columns = df.columns.str.lower()
df.rename(columns={
    'patientid': 'patient_id',
    'appointmentid': 'appointment_id',
    'scheduledday': 'scheduled_day',
    'appointmentday': 'appointment_day',
    'hipertension': 'hypertension',
    'handcap': 'handicap',
    'no-show': 'no_show'
}, inplace=True)

# 5. Convert dates to datetime
df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
df['appointment_day'] = pd.to_datetime(df['appointment_day'])

# 6. Handle age outliers
df['age'] = df['age'].apply(lambda x: 0 if x < 0 else x)
df['age'] = df['age'].apply(lambda x: 110 if x > 110 else x)

# 7. Create new features
df['days_between'] = (df['appointment_day'] - df['scheduled_day']).dt.days
df['days_between'] = df['days_between'].apply(lambda x: 0 if x < 0 else x)

# Age groups
bins = [0, 12, 19, 30, 50, 65, 110]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Day of week features
df['appointment_dow'] = df['appointment_day'].dt.day_name()
df['scheduled_dow'] = df['scheduled_day'].dt.day_name()

# 8. Ensure correct data types
numeric_cols = ['scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handicap', 'sms_received']
df[numeric_cols] = df[numeric_cols].astype('int8')
df['age'] = df['age'].astype('int8')

# 9. Final check
print("\nData types after cleaning:")
print(df.dtypes)

print("\nSample of cleaned data:")
print(df.head())

# Save cleaned dataset
df.to_csv('cleaned_noshowappointments.csv', index=False)