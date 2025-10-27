import os 
import pandas as pd
import glob

#find current path

current_path = os.path.dirname(os.path.abspath(__file__))

new_path = os.path.join(current_path, 'questio-merge')

og_path = os.path.join(current_path, 'sourcedata')

#from new path open each csv file one by one

new_sub_number = 'sub-052'

iasta1 = glob.glob(os.path.join(new_path, 'IASTA1.csv'))
df_iasta1 = pd.read_csv(iasta1[0])
df_iasta1 = df_iasta1[-1:]
df_iasta1['#Participant'] = new_sub_number
df_iasta1.to_csv(iasta1[0], index=False)

    

iasta2 = glob.glob(os.path.join(new_path, 'IASTA2.csv'))
df_iasta2 = pd.read_csv(iasta2[0])
df_iasta2 = df_iasta2[-1:]
df_iasta2['Numéro participant'] = new_sub_number
#drop column #Participant if it exists
if '#Participant' in df_iasta2.columns:
    df_iasta2 = df_iasta2.drop(columns=['#Participant'])

df_iasta2.to_csv(iasta2[0], index=False)


PCS = glob.glob(os.path.join(new_path, 'PCS.csv'))
df_PCS = pd.read_csv(PCS[0])
df_PCS = df_PCS[-1:]
df_PCS['# du participant:'] = new_sub_number

socio = glob.glob(os.path.join(new_path, 'socio.csv'))
df_socio = pd.read_csv(socio[0])
df_socio = df_socio[-1:]

#lets go open the orginal source data questionnaires and then add our new data to them

ista_y1 = glob.glob(os.path.join(og_path, 'iasta_y1.csv'))
df_ista_y1 = pd.read_csv(ista_y1[0])
df_ista_y1 = pd.concat([df_ista_y1, df_iasta1], ignore_index=True)
#save back to csv under og_path
df_ista_y1.to_csv(ista_y1[0], index=False)

ista_y2 = glob.glob(os.path.join(og_path, 'iasta_y2.csv'))
df_ista_y2 = pd.read_csv(ista_y2[0])
df_ista_y2 = pd.concat([df_ista_y2, df_iasta2], ignore_index=True)
#save back to csv under og_path
df_iasta2 = df_iasta2[:-1]
df_ista_y2.to_csv(ista_y2[0], index=False)




pcs = glob.glob(os.path.join(og_path, 'pcs.csv'))
df_pcs = pd.read_csv(pcs[0])
df_pcs = pd.concat([df_pcs, df_PCS], ignore_index=True)
#save back to csv under og_path
df_pcs.to_csv(pcs[0], index=False)


# Helper function to remove duplicates
def remove_duplicate_subs(file_path, sub_col):
    df = pd.read_csv(file_path)
    print(f"Before cleanup: {len(df)} rows in {os.path.basename(file_path)}")

    # Drop duplicates by participant column, keeping only the last occurrence
    df = df.drop_duplicates(subset=[sub_col], keep='last')

    df.to_csv(file_path, index=False)
    print(f"After cleanup: {len(df)} rows\n")

# Run for each relevant file
remove_duplicate_subs(os.path.join(og_path, 'iasta_y1.csv'), '#Participant')
remove_duplicate_subs(os.path.join(og_path, 'iasta_y2.csv'), 'Numéro participant')
remove_duplicate_subs(os.path.join(og_path, 'pcs.csv'), '# du participant:')

