"""
05. Format damien sepsis data
=============================

Example
"""
# Libraries
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# Constants
# ---------------------------
# Path to biochemical markers
path_bio = '.\\datasets\\damien-sepsis-biomarkers.csv'

# Path to nhs to hos mappings
path_nth = '.\\datasets\\damien-sepsis-nhs_to_hos.csv'

# Path to data request megalist
path_drm = '.\\datasets\\data-request-megalist.xlsx'

# Path to save output
path_save = '.\\outputs\\{0}-damien-sepsis-biomarkers-pm{1}.csv'

# Save
SAVE = True

# Days +- first micro sample
WINDOW = 30

# ---------------------------
# Main
# ---------------------------

# -----------
# Read data
# -----------
# Read biomarkers
bio = pd.read_csv(path_bio,
	#nrows=10000,
	parse_dates=['date_collected',
			     'date_outcome'])

# Read nhs to hos
nth = pd.read_csv(path_nth)

# Read data request megalist
drm = pd.read_excel(path_drm,
	parse_dates=['Sampledate'])

# Rename drm
drm = drm.rename(columns={
	'Sampledate': 'date_sample',
	'Hospital Number': 'hos_number'})

# Sort by date (important if keeping first)
drm = drm.sort_values(by='date_sample')

# Keep first appearance only
drm = drm.groupby(by='hos_number') \
		 .first().reset_index()

# Show
print("\nShow datasets:")
print(bio)
print(nth)
print(drm)

# Show columns
print("\nShow columns:")
print(bio.columns)
print(nth.columns)
print(drm.columns)

# -----------
# Merge
# -----------
# Merge by nhs_number
bio = bio.merge(nth, how='left',
	left_on='patient_nhs_number',
    right_on='nhs_number')

# Merge with date (first)
bio = bio.merge(drm, how='inner',
	left_on='hos_number',
	right_on='hos_number')

# .. note: There must be an issue with Sampledate, because it is not
#          being converted to datetime64[ns] from parse_dates. Thus
#          force conversion ourselves. Note that invalid parsing will
#          be set to NaT (not a time)
bio.date_sample = \
	pd.to_datetime(bio.date_sample, errors='coerce')

# Compute day difference
bio['day'] = (bio.date_sample - bio.date_collected).dt.days

# -----------
# Plot
# -----------
# Count
count = bio.day.value_counts().sort_index()

# Configure sns
#sns.set_theme(style='whitegrid')
sns.set_color_codes("muted")
sns.despine(left=True, bottom=True)

# Plot bars
ax = plt.bar(count.index.values,
	count.values, color='b', alpha=0.5)

# Fill aea selected
plt.fill_between(x=[-WINDOW, WINDOW],
	y1=0, y2=count.max(), alpha=0.25,
	color='orange')

# Draw vertical line at 30
plt.vlines([-WINDOW, WINDOW], ymin=0,
	ymax=count.max(), color='k',
	linestyle='dashed', linewidth=0.75)

# Configure
plt.grid(False)
plt.xlabel('Day from sample')
plt.ylabel('Count')
plt.title('Day from sample count')

# Layout
plt.tight_layout()

# Show
plt.show()

# ---------------
# Filter and save
# ---------------
# Filter out
bio = bio[bio.day.abs() <= WINDOW]

# Save
if SAVE:
	# Get time
	time = time.strftime('%Y%m%d-%H%M%S')
	# Save with all info
	bio.to_csv(path_save.format(time, str(WINDOW)))
	# Save anonymised
	bio = bio.drop(columns=['patient_nhs_number',
							'nhs_number',
							'hos_number'])
	# Show columns
	print(bio.columns)
	bio.to_csv(path_save.format(time, str(WINDOW) + '-anonymised'))

# Show
plt.show()