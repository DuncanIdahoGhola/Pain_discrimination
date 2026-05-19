# %%
import pandas as pd
import numpy as np 
import os

file_path = r'C:\Users\ANCYB2\Documents\GitHub\Pain_discrimination\sourcedata'

# Load files
iasta_y1 = pd.read_csv(os.path.join(file_path, 'iasta_y1.csv'))
iasta_y2 = pd.read_csv(os.path.join(file_path, 'iasta_y2.csv'))
pcs = pd.read_csv(os.path.join(file_path, 'pcs.csv'))
sociodemo = pd.read_csv(os.path.join(file_path, 'sociodemo.csv'))

# Set participant columns as index
iasta_y1.set_index('#Participant', inplace=True)
iasta_y2.set_index('Numéro participant', inplace=True)
pcs.set_index('# du participant:', inplace=True)
sociodemo.set_index('  # Participant  ', inplace=True)

# Merge all dataframes
df = pd.concat([iasta_y1, iasta_y2, pcs, sociodemo], axis=1)

# ==========================================
# CREATE A SEPARATE RESULTS DATAFRAME
# ==========================================
#empty results df

results_df = pd.DataFrame(index=df.index)

for p in results_df.index:

    # --------------------------
    # IASTA Y2
    # --------------------------
    if p in iasta_y2.index:

        all_iasta2 = []

        for c in range(1, len(iasta_y2.columns)):

            try:
                value = int(str(iasta_y2.loc[p, iasta_y2.columns[c]])[0])

                results_df.loc[p, "qiastay2_" + iasta_y2.columns[c]] = value
                all_iasta2.append(value)

            except:
                all_iasta2.append(np.nan)

        if len(all_iasta2) == 20:

            all_iasta2 = np.asarray(all_iasta2)

            reverse_idx = [0, 2, 5, 6, 9, 12, 13, 15, 18]

            all_iasta2[reverse_idx] = 5 - all_iasta2[reverse_idx]

            results_df.loc[p, "iastay2_total"] = np.nansum(all_iasta2)

    # --------------------------
    # IASTA Y1
    # --------------------------
    if p in iasta_y1.index:

        all_iasta1 = []

        for c in range(1, len(iasta_y1.columns)):

            try:
                value = int(str(iasta_y1.loc[p, iasta_y1.columns[c]])[0])

                results_df.loc[p, "qiastay1_" + iasta_y1.columns[c]] = value
                all_iasta1.append(value)

            except:
                all_iasta1.append(np.nan)

        if len(all_iasta1) == 20:

            all_iasta1 = np.asarray(all_iasta1)

            reverse_idx = [0, 1, 4, 7, 9, 10, 14, 15, 18, 19]

            all_iasta1[reverse_idx] = 5 - all_iasta1[reverse_idx]

            results_df.loc[p, "iastay1_total"] = np.nansum(all_iasta1)

    # --------------------------
    # PCS
    # --------------------------
    if p in pcs.index:

        all_pcs = []

        for c in range(1,len(pcs.columns)):

            try:
                value = int(str(pcs.loc[p, pcs.columns[c]])[0])

                results_df.loc[p, "qpcs_" + pcs.columns[c]] = value
                all_pcs.append(value)

            except:
                all_pcs.append(np.nan)

        if len(all_pcs) == 13:

            results_df.loc[p, "pcs_total"] = np.nansum(all_pcs)


# ==========================================
# ADD SOCIODEMO VARIABLES TO results_df
# ==========================================

# Initialize columns
results_df["age"] = np.nan
results_df["ismale"] = 0
results_df["isfemale"] = 0
results_df["Autres"] = 0

for row in sociodemo.iterrows():

    participant = row[0]

    if participant in results_df.index:

        # Age
        try:
            results_df.loc[participant, "age"] = int(
                row[1]["2. Quel est votre âge en années? "]
            )
        except:
            results_df.loc[participant, "age"] = np.nan

        # Gender
        gender = row[1]["4. Quel est votre genre? "]

        results_df.loc[participant, "ismale"] = int(gender == "Masculin")
        results_df.loc[participant, "isfemale"] = int(gender == "Féminin")
        results_df.loc[participant, "Autres"] = int(gender == "Autres")

# Convert to integer type
results_df["ismale"] = results_df["ismale"].astype(int)
results_df["isfemale"] = results_df["isfemale"].astype(int)
results_df["Autres"] = results_df["Autres"].astype(int)






# Save results dataframe with just the total scores
results_df = results_df[["iastay1_total", "iastay2_total", "pcs_total", 'age', 'ismale', 'isfemale', 'Autres']]

results_df = results_df.reset_index()
results_df.rename(columns={"index": "participant"}, inplace=True)

results_df.to_csv(
    os.path.join(file_path, "results_df.csv"),
    index=False
)
